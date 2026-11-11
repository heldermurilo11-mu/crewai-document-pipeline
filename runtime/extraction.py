"""Hardened high-DPI rasterized document extraction via Gemini Vision."""

import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Callable

from google import genai
from google.genai import types
import pymupdf

from tools.orientation_tool import OrientationTool
from .deadline import Deadline, DeadlineExceeded
from .files import atomic_json

logger = logging.getLogger(__name__)

UPLOAD_TTL_SECONDS = 47 * 3600
MAX_STREAM_BYTES = 32 * 1024 * 1024

EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "required": ["covered_pages", "samples"],
    "properties": {
        "has_printed_header_id": {"type": "BOOLEAN"},
        "collection_receipt_timestamp": {"type": "STRING", "nullable": True},
        "covered_pages": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "samples": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["codigo", "identificacao", "data", "hora", "source_pages"],
                "properties": {
                    "template": {"type": "STRING", "nullable": True},
                    "codigo": {"type": "STRING"},
                    "identificacao": {"type": "STRING", "nullable": True},
                    "data": {"type": "STRING", "nullable": True},
                    "hora": {"type": "STRING", "nullable": True},
                    "source_pages": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                },
            },
        },
    },
}


class ExtractionRejected(ValueError):
    """Raised when model output fails schema, completeness, or coverage invariants."""


def http_options(deadline: Deadline, configured_ms: int) -> types.HttpOptions:
    return types.HttpOptions(
        timeout=deadline.timeout_ms(configured_ms),
        retry_options=types.HttpRetryOptions(attempts=1),
    )


def transient(error: BaseException) -> bool:
    if isinstance(error, (DeadlineExceeded, KeyboardInterrupt, ExtractionRejected)):
        return False
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    resp = getattr(error, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    if isinstance(status, int) and status in (408, 429, 500, 502, 503, 504):
        return True
    message = str(error).lower()
    return any(
        token in message
        for token in ("429", "500", "502", "503", "504", "timeout", "timed out", "temporar", "unavailable")
    )


def _finish_reason_name(candidate) -> str:
    reason = getattr(candidate, "finish_reason", None)
    if reason is None:
        return ""
    name = getattr(reason, "name", None)
    return str(name or reason).split(".")[-1].upper()


def complete_text(
    client,
    model: str,
    contents,
    config,
    *,
    streaming: bool = False,
    deadline: Deadline | None = None,
) -> tuple[str, str, object]:
    """Collect model output requiring an explicit ``STOP`` finish reason."""
    if streaming and hasattr(client.models, "generate_content_stream"):
        parts: list[str] = []
        total_bytes = 0
        finish_reason = ""
        model_version = model
        usage = None
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if deadline is not None:
                deadline.check("streaming chunk")
            text = getattr(chunk, "text", None) or ""
            if text:
                total_bytes += len(text.encode("utf-8"))
                if total_bytes > MAX_STREAM_BYTES:
                    raise ExtractionRejected("Streaming response exceeded 32 MB safety limit")
                parts.append(text)
            for cand in getattr(chunk, "candidates", None) or []:
                fr = _finish_reason_name(cand)
                if fr:
                    finish_reason = fr
            model_version = getattr(chunk, "model_version", None) or model_version
            usage = getattr(chunk, "usage_metadata", None) or usage
        if finish_reason != "STOP":
            raise ExtractionRejected(f"Incomplete streamed extraction (finish_reason={finish_reason or 'NONE'})")
        return "".join(parts), model_version, usage

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    candidates = getattr(response, "candidates", None) or []
    finish_reason = _finish_reason_name(candidates[0]) if candidates else ""
    if finish_reason != "STOP":
        raise ExtractionRejected(f"Truncated extraction response (finish_reason={finish_reason or 'NONE'})")
    return (getattr(response, "text", None) or ""), getattr(response, "model_version", model), getattr(
        response, "usage_metadata", None
    )


def validate_coverage(extraction: dict, total_pages: int) -> None:
    """Enforce full 1..N page coverage and per-sample page provenance."""
    expected = set(range(1, total_pages + 1))
    covered_raw = extraction.get("covered_pages") or []
    try:
        covered = {int(page) for page in covered_raw}
    except (TypeError, ValueError) as exc:
        raise ExtractionRejected("Invalid page numbers in covered_pages") from exc
    if covered != expected:
        missing = sorted(expected - covered)
        raise ExtractionRejected(f"Missing page coverage for pages: {missing}")

    samples = extraction.get("samples") or []
    for index, sample in enumerate(samples):
        pages = sample.get("source_pages")
        if not isinstance(pages, list) or not pages:
            raise ExtractionRejected(f"Sample #{index + 1} is missing source_pages provenance")
        for page in pages:
            if not isinstance(page, int) or page < 1 or page > total_pages:
                raise ExtractionRejected(f"Sample #{index + 1} references invalid page {page}")


def _flatten_extraction(dados_raw) -> tuple[list[dict], str, bool, list[int]]:
    samples: list[dict] = []
    receipt_ts = ""
    has_printed_header_id = False
    covered_pages: list[int] = []

    if isinstance(dados_raw, dict):
        samples = list(dados_raw.get("samples", []))
        receipt_ts = str(dados_raw.get("collection_receipt_timestamp", "") or "").strip()
        has_printed_header_id = bool(dados_raw.get("has_printed_header_id", False))
        covered_pages = list(dados_raw.get("covered_pages", []))
        if not samples and (dados_raw.get("codigo") or dados_raw.get("identificacao")):
            samples = [dados_raw]
    elif isinstance(dados_raw, list):
        for item in dados_raw:
            if not isinstance(item, dict):
                continue
            if "samples" in item:
                samples.extend(item.get("samples", []))
                if item.get("covered_pages"):
                    covered_pages.extend(item["covered_pages"])
                if not receipt_ts and item.get("collection_receipt_timestamp"):
                    receipt_ts = str(item["collection_receipt_timestamp"]).strip()
                if item.get("has_printed_header_id"):
                    has_printed_header_id = True
            else:
                samples.append(item)
    return samples, receipt_ts, has_printed_header_id, sorted(set(covered_pages))


def _merge_sample_continuations(samples: list[dict]) -> list[dict]:
    """Merge multi-page sample continuations cleanly and reject attribute conflicts."""
    merged: list[dict] = []
    by_code: dict[str, dict] = {}
    for sample in samples:
        code = sample.get("codigo", "")
        if not code:
            merged.append(sample)
            continue
        existing = by_code.get(code)
        if existing is None:
            by_code[code] = sample
            merged.append(sample)
            continue
        for field in ("identificacao", "data", "hora"):
            cur = existing.get(field, "")
            nxt = sample.get(field, "")
            if nxt and cur and nxt != cur:
                raise ExtractionRejected(
                    f"Conflicting {field} across pages for sample {code}: '{cur}' vs '{nxt}'"
                )
            if nxt and not cur:
                existing[field] = nxt
        existing["source_pages"] = sorted(
            set(existing.get("source_pages", [])) | set(sample.get("source_pages", []))
        )
    return merged


def _upload_images(
    client,
    image_paths: list[Path],
    cache_path: Path,
    deadline: Deadline,
) -> list:
    now = time.time()
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}

    # Prune expired cache entries beyond 47h Gemini Files API TTL
    cache = {k: v for k, v in cache.items() if now - v.get("uploaded_at", 0) < UPLOAD_TTL_SECONDS}
    parts = []
    cache_dirty = False

    for path in image_paths:
        deadline.check("file upload")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cached = cache.get(digest)
        uploaded_ref = None
        if cached and cached.get("uri") and hasattr(client.files, "get"):
            try:
                uploaded_ref = client.files.get(name=cached["name"])
            except Exception:
                uploaded_ref = None
        if uploaded_ref is None:
            uploaded_ref = client.files.upload(file=str(path))
            cache[digest] = {
                "name": getattr(uploaded_ref, "name", str(path.name)),
                "uri": getattr(uploaded_ref, "uri", None),
                "uploaded_at": now,
            }
            cache_dirty = True
        parts.append(uploaded_ref)

    if cache_dirty:
        atomic_json(cache_path, cache)
    return parts


def _render_pdf_pages(pdf_path: Path, workspace: Path, dpi: int = 300) -> list[Path]:
    """Rasterize PDF pages as lossless PNGs without modifying the source PDF."""
    workspace.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    with pymupdf.open(str(pdf_path)) as doc:
        if doc.page_count < 1:
            raise ExtractionRejected(f"Empty PDF document: {pdf_path.name}")
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            out_path = workspace / f"page-{idx:04d}.png"
            pix.save(str(out_path))
            png_paths.append(out_path)
    return png_paths


def _rotate_pages_from_source(
    source_path: Path,
    page_paths: list[Path],
    rotations: list[int],
    *,
    is_pdf: bool,
    dpi: int = 300,
) -> None:
    """Re-render ONLY rotated pages upright in memory without mutating *source_path*."""
    if is_pdf:
        with pymupdf.open(str(source_path)) as doc:
            for idx, (out_path, angle) in enumerate(zip(page_paths, rotations)):
                if angle in (90, 180, 270):
                    page = doc[idx]
                    page.set_rotation((page.rotation + angle) % 360)
                    page.get_pixmap(dpi=dpi, alpha=False).save(str(out_path))
        return

    for out_path, angle in zip(page_paths, rotations):
        if angle in (90, 180, 270):
            with pymupdf.open(str(out_path)) as img_doc:
                page = img_doc[0]
                page.set_rotation((page.rotation + angle) % 360)
                page.get_pixmap(alpha=False).save(str(out_path))


def extract_document(
    file_path: str | Path,
    api_key: str,
    prompt: str,
    config,
    deadline: Deadline,
    normalize_code_fn: Callable[[str], str],
    normalize_time_fn: Callable[[str], str],
    *,
    work_dir: str | Path | None = None,
    template_name: str = "GENERIC",
    match_method: str = "unknown",
    confidence: float = 0.0,
) -> dict:
    """Render pages to 300 DPI lossless PNGs, correct orientation, and run hardened Gemini extraction."""
    source_path = Path(file_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Document not found: {source_path}")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for document extraction")

    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    temp_ctx = None if work_dir is not None else tempfile.TemporaryDirectory(prefix="docpipe-")
    workspace = Path(work_dir) if work_dir is not None else Path(temp_ctx.name)
    workspace.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    try:
        # Step 1: Strictly render source document BEFORE touching Files API uploads.
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            page_paths = _render_pdf_pages(source_path, workspace, dpi=300)
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            out_path = workspace / "page-0001.png"
            with pymupdf.open(str(source_path)) as img_doc:
                img_doc[0].get_pixmap(alpha=False).save(str(out_path))
            page_paths = [out_path]
        else:
            raise ExtractionRejected(f"Unsupported file type: {suffix}")

        total_pages = len(page_paths)

        # Step 2: Optional per-page orientation correction (without altering original PDF).
        if getattr(config, "orientation_enabled", True):
            detector = OrientationTool(api_key=api_key, model=config.gemini_model_fast)
            rotations = detector.detect_images(
                client,
                page_paths,
                deadline=deadline,
                request_timeout_ms=min(120_000, config.gemini_request_timeout_ms),
            )
            if any(angle in (90, 180, 270) for angle in rotations):
                logger.info("Applying per-page orientation corrections: %s", rotations)
                _rotate_pages_from_source(
                    source_path,
                    page_paths,
                    rotations,
                    is_pdf=(suffix == ".pdf"),
                    dpi=300,
                )

        # Step 3: Upload PNG images once with SHA-256 47h TTL cache.
        cache_path = workspace / "uploads.json"
        file_parts = _upload_images(client, page_paths, cache_path, deadline)

        full_prompt = (
            f"{prompt}\n\n"
            f"### COVERAGE & PROVENANCE REQUIREMENTS\n"
            f"- The document has exactly {total_pages} page(s) (1..{total_pages}).\n"
            f"- Return `covered_pages`: [1, ..., {total_pages}] confirming every page was read.\n"
            f"- Include `source_pages`: [page_numbers] on EVERY extracted sample.\n"
        )
        contents = [*file_parts, full_prompt]

        last_error: Exception | None = None
        for attempt in range(config.extraction_max_attempts):
            try:
                deadline.check("extraction request")
                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EXTRACTION_SCHEMA,
                    temperature=0.1,
                    http_options=http_options(deadline, config.gemini_request_timeout_ms),
                )
                raw_text, model_version, _usage = complete_text(
                    client,
                    config.gemini_model_pro,
                    contents,
                    gen_config,
                    streaming=getattr(config, "streaming_enabled", True),
                    deadline=deadline,
                )
                parsed = json.loads(raw_text)
                raw_samples, receipt_ts, has_printed_header_id, covered_pages = _flatten_extraction(parsed)
                if not covered_pages:
                    covered_pages = []
                validate_coverage(
                    {"covered_pages": covered_pages, "samples": raw_samples},
                    total_pages,
                )

                normalized_samples: list[dict] = []
                for item in raw_samples:
                    sample = {
                        "template": str(item.get("template") or template_name).strip(),
                        "codigo": normalize_code_fn(item.get("codigo", "") or ""),
                        "identificacao": (
                            str(item.get("identificacao", "") or "")
                            .replace(";", ",")
                            .replace("\n", " ")
                            .strip()
                        ),
                        "data": str(item.get("data", "") or "").strip(),
                        "hora": normalize_time_fn(str(item.get("hora", "") or "")),
                        "source_pages": [int(p) for p in item.get("source_pages", [])],
                    }
                    if sample["codigo"] or sample["identificacao"]:
                        normalized_samples.append(sample)

                merged_samples = _merge_sample_continuations(normalized_samples)
                return {
                    "template": template_name,
                    "file": source_path.name,
                    "source_sha256": source_sha256,
                    "image_page_count": total_pages,
                    "covered_pages": covered_pages,
                    "has_printed_header_id": has_printed_header_id,
                    "collection_receipt_timestamp": receipt_ts,
                    "model_version": model_version,
                    "match_method": match_method,
                    "confidence": confidence,
                    "samples": merged_samples,
                    "total_extracted": len(merged_samples),
                }
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= config.extraction_max_attempts or not transient(exc):
                    raise
                deadline.sleep(min(2.0 * (attempt + 1), 10.0))

        assert last_error is not None
        raise last_error
    finally:
        if hasattr(client, "close") and callable(client.close):
            try:
                client.close()
            except Exception:
                pass
        if temp_ctx is not None:
            temp_ctx.cleanup()
