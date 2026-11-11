"""Offline tests for 300 DPI PDF rasterization, orientation, upload cache, and schema guards."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf

from runtime.config import RuntimeConfig
from runtime.deadline import Deadline
import runtime.extraction as extraction_mod
from runtime.extraction import (
    ExtractionRejected,
    complete_text,
    http_options,
    validate_coverage,
)
from tools.filler_agent import FillerAgentTool


def make_pdf(path: Path, page_count: int = 2, *, rotate_first: bool = False) -> None:
    doc = pymupdf.open()
    for idx in range(page_count):
        page = doc.new_page(width=144, height=144)  # 2x2 inches -> 600x600 px at 300 DPI
        page.insert_text((18, 48), f"COC #{idx + 1}")
        if rotate_first and idx == 0:
            page.set_rotation(90)
    doc.save(str(path))
    doc.close()


class FakeFilesAPI:
    def __init__(self, uploads: list[Path], upload_meta: list[tuple[int, int, int, int]]):
        self.uploads = uploads
        self.upload_meta = upload_meta
        self.lookup_calls = 0

    def upload(self, *, file: str):
        path = Path(file)
        self.uploads.append(path)
        pix = pymupdf.Pixmap(str(path))
        self.upload_meta.append((pix.xres, pix.yres, pix.width, pix.height))
        return SimpleNamespace(
            name=f"files/{len(self.uploads)}",
            uri=f"https://example.invalid/files/{len(self.uploads)}",
        )

    def get(self, *, name: str):
        self.lookup_calls += 1
        return SimpleNamespace(name=name, uri=f"https://example.invalid/{name}")


class FakeModelsAPI:
    def __init__(self, parent):
        self.parent = parent

    def generate_content(self, *, model, contents, config):
        self.parent.contents.append(contents)
        schema = getattr(config, "response_schema", None)
        if schema and "rotations" in schema.get("properties", {}):
            payload = {"rotations": self.parent.rotations}
        else:
            payload = {
                "has_printed_header_id": False,
                "collection_receipt_timestamp": "10/02/2026 09:00",
                "covered_pages": list(range(1, self.parent.pages + 1)),
                "samples": [
                    {
                        "template": "LAB_ALPHA",
                        "codigo": "1823875",
                        "identificacao": "PONTO NORTE",
                        "data": "10/02/2026",
                        "hora": "09:45",
                        "source_pages": [1],
                    }
                ],
            }
        return SimpleNamespace(
            text=json.dumps(payload),
            candidates=[SimpleNamespace(finish_reason="STOP")],
            model_version=model,
            usage_metadata=None,
        )

    def generate_content_stream(self, *, model, contents, config):
        resp = self.generate_content(model=model, contents=contents, config=config)
        yield SimpleNamespace(
            text=resp.text,
            candidates=resp.candidates,
            model_version=resp.model_version,
            usage_metadata=None,
        )


class FakeClient:
    def __init__(self, pages: int = 2, rotations: list[int] | None = None):
        self.pages = pages
        self.rotations = rotations or [0] * pages
        self.uploads: list[Path] = []
        self.upload_meta: list[tuple[int, int, int, int]] = []
        self.contents: list = []
        self.closed = False
        self.files = FakeFilesAPI(self.uploads, self.upload_meta)
        self.models = FakeModelsAPI(self)

    def close(self):
        self.closed = True


class ExtractionTests(unittest.TestCase):
    def test_source_pdf_is_untouched_and_pngs_go_to_work_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "document.pdf"
            source.parent.mkdir()
            make_pdf(source, 2)
            original = source.read_bytes()
            workspace = root / "output" / "pages"
            client = FakeClient(pages=2)
            config = replace(RuntimeConfig(), orientation_enabled=False)
            with patch("runtime.extraction.genai.Client", return_value=client):
                result = FillerAgentTool(api_key="fake", runtime_config=config).extract(
                    str(source), work_dir=workspace
                )
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(source.parent.iterdir()), [source])
            self.assertEqual(len(list(workspace.glob("*.png"))), 2)
            self.assertEqual(result["total_extracted"], 1)
            self.assertTrue(client.closed)

    def test_transport_timeout_unit_is_ms_and_clamped(self):
        options = http_options(Deadline(0.5), 900_000)
        self.assertGreater(options.timeout, 0)
        self.assertLessEqual(options.timeout, 500)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_missing_page_coverage_is_rejected(self):
        with self.assertRaises(ExtractionRejected):
            validate_coverage({"covered_pages": [1], "samples": [{"source_pages": [1]}]}, 2)

    def test_missing_sample_provenance_is_rejected(self):
        with self.assertRaises(ExtractionRejected):
            validate_coverage({"covered_pages": [1, 2], "samples": [{}]}, 2)

    def test_corrupt_pdf_fails_before_upload_and_closes_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.pdf"
            path.write_bytes(b"not a PDF")
            client = FakeClient()
            config = replace(RuntimeConfig(), orientation_enabled=False)
            with patch("runtime.extraction.genai.Client", return_value=client), self.assertRaises(
                Exception
            ):
                FillerAgentTool(api_key="fake", runtime_config=config).extract(str(path))
            self.assertEqual(client.uploads, [])
            self.assertTrue(client.closed)

    def test_52_pages_remain_300_dpi_png_in_single_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-52.pdf"
            make_pdf(path, 52)
            client = FakeClient(pages=52)
            config = replace(RuntimeConfig(), orientation_enabled=False)
            with patch("runtime.extraction.genai.Client", return_value=client):
                result = FillerAgentTool(api_key="fake", runtime_config=config).extract(str(path))
            self.assertEqual(result["image_page_count"], 52)
            self.assertEqual(result["covered_pages"], list(range(1, 53)))
            self.assertEqual(len(client.contents), 1)
            refs = [part for part in client.contents[0] if not isinstance(part, str)]
            self.assertEqual(len(refs), 52)
            self.assertEqual(len(client.upload_meta), 52)
            for xres, yres, width, height in client.upload_meta:
                self.assertEqual((xres, yres), (300, 300))
                self.assertEqual((width, height), (600, 600))
            self.assertTrue(client.closed)

    def test_mixed_orientation_rotates_only_needed_page_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mixed.pdf"
            work_dir = root / "work"
            make_pdf(path, 2, rotate_first=True)
            client = FakeClient(pages=2, rotations=[270, 0])
            config = RuntimeConfig()
            with patch("runtime.extraction.genai.Client", return_value=client):
                result = FillerAgentTool(api_key="fake", runtime_config=config).extract(
                    str(path), work_dir=work_dir
                )
            self.assertEqual(result["image_page_count"], 2)
            self.assertEqual(len(client.contents), 2)  # 1 batched orientation + 1 extraction
            with pymupdf.open(str(path)) as original:
                self.assertEqual(original[0].rotation, 90)
                self.assertEqual(original[1].rotation, 0)
            for png in work_dir.glob("*.png"):
                pix = pymupdf.Pixmap(str(png))
                self.assertEqual((pix.xres, pix.yres), (300, 300))

    def test_truncated_response_is_rejected(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_: SimpleNamespace(
                    text='{"covered_pages":[1],"samples":[]}',
                    candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
                )
            )
        )
        with self.assertRaises(ExtractionRejected):
            complete_text(client, "model", [], None)

    def test_partial_stream_is_rejected_and_complete_stop_stream_assembles(self):
        partial_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=lambda **_: iter(
                    [SimpleNamespace(text='{"covered_pages":', candidates=[], model_version="m", usage_metadata=None)]
                )
            )
        )
        with self.assertRaises(ExtractionRejected):
            complete_text(partial_client, "m", [], None, streaming=True, deadline=Deadline(10))

        full_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=lambda **_: iter(
                    [
                        SimpleNamespace(text='{"covered_pages":[1],', candidates=[], model_version="m", usage_metadata=None),
                        SimpleNamespace(
                            text='"samples":[]}',
                            candidates=[SimpleNamespace(finish_reason="STOP")],
                            model_version="m",
                            usage_metadata=None,
                        ),
                    ]
                )
            )
        )
        text, model_v, _ = complete_text(
            full_client, "m", [], None, streaming=True, deadline=Deadline(10)
        )
        self.assertEqual(model_v, "m")
        self.assertEqual(json.loads(text), {"covered_pages": [1], "samples": []})

    def test_stream_byte_limit_is_enforced(self):
        with patch.object(extraction_mod, "MAX_STREAM_BYTES", 32):
            big_client = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=lambda **_: iter(
                        [
                            SimpleNamespace(
                                text="x" * 64,
                                candidates=[SimpleNamespace(finish_reason="STOP")],
                                model_version="m",
                                usage_metadata=None,
                            )
                        ]
                    )
                )
            )
            with self.assertRaises(ExtractionRejected):
                complete_text(big_client, "m", [], None, streaming=True)

    def test_upload_cache_reuses_uploaded_uris_within_ttl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cached.pdf"
            work_dir = root / "work"
            make_pdf(source, 1)
            config = replace(RuntimeConfig(), orientation_enabled=False)

            first = FakeClient(pages=1)
            with patch("runtime.extraction.genai.Client", return_value=first):
                FillerAgentTool(api_key="fake", runtime_config=config).extract(
                    str(source), work_dir=work_dir
                )
            self.assertEqual(len(first.uploads), 1)
            self.assertTrue((work_dir / "uploads.json").is_file())

            second = FakeClient(pages=1)
            with patch("runtime.extraction.genai.Client", return_value=second):
                FillerAgentTool(api_key="fake", runtime_config=config).extract(
                    str(source), work_dir=work_dir
                )
            self.assertEqual(len(second.uploads), 0)
            self.assertEqual(second.files.lookup_calls, 1)

    def test_prompt_uses_configurable_prefixes_without_hardcodes(self):
        prompt = FillerAgentTool(expected_code_prefixes=["41", "42"])._build_extraction_prompt(
            "LAB_ALPHA",
            {"codigo": {"location": "coluna Codigo"}},
            ["Regra configuravel"],
        )
        self.assertIn("LAB_ALPHA", prompt)
        self.assertIn("'41'", prompt)
        self.assertIn("'42'", prompt)
        self.assertIn("Regra configuravel", prompt)


if __name__ == "__main__":
    unittest.main()
