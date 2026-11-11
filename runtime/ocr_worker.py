"""Isolated child-process extraction worker executed via ``runtime.supervisor.run_worker``."""

import hashlib
import os
from pathlib import Path

from runtime.config import RuntimeConfig
from runtime.deadline import Deadline
from runtime.files import atomic_json
from tools.drive_tools import GoogleDriveDownloadTool
from tools.filler_agent import FillerAgentTool
from tools.pattern_identifier import PatternIdentifierTool


def _md5_hex(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(
    *,
    file_path: str,
    api_key: str,
    config_values: dict,
    seconds: float,
    progress_path: str,
    expected_md5: str | None = None,
    download: dict | None = None,
    allow_template_inference: bool = True,
) -> dict:
    """Worker entrypoint executed in a spawned, OS-contained child process."""
    deadline = Deadline(seconds)
    config = RuntimeConfig(**config_values)
    config.validate()
    target = Path(file_path)
    progress = Path(progress_path)

    atomic_json(progress, {"stage": "starting", "file": target.name})

    if download and not target.exists():
        deadline.check("drive download")
        atomic_json(progress, {"stage": "downloading", "file": target.name})
        drive = GoogleDriveDownloadTool(
            credentials_dict=download.get("credentials"),
            token_dict=download.get("token"),
            deadline=deadline,
            request_timeout=config.drive_request_timeout_seconds,
        )
        drive.download_file(download["id"], download["name"], destination=target)

    if not target.is_file():
        raise FileNotFoundError(f"Worker input file missing: {target}")

    if expected_md5:
        actual_md5 = _md5_hex(target)
        if actual_md5.lower() != expected_md5.lower():
            raise ValueError(
                f"Source checksum mismatch for {target.name}: expected={expected_md5} actual={actual_md5}"
            )

    deadline.check("template identification")
    atomic_json(progress, {"stage": "identifying", "file": target.name})
    identifier = PatternIdentifierTool(api_key=api_key or os.getenv("GEMINI_API_KEY"))
    identification = identifier.identify(
        str(target),
        deadline=deadline,
        allow_inference=allow_template_inference,
    )
    if "error" in identification:
        raise RuntimeError(identification["error"])

    deadline.check("document extraction")
    atomic_json(
        progress,
        {"stage": "extracting", "file": target.name, "template": identification.get("template_name")},
    )
    work_pages_dir = target.parent / "pages"
    filler = FillerAgentTool(
        api_key=api_key or os.getenv("GEMINI_API_KEY"),
        max_retries=config.extraction_max_attempts,
        runtime_config=config,
    )
    result = filler.extract(
        str(target),
        identification_result=identification,
        deadline=deadline,
        work_dir=work_pages_dir,
    )
    atomic_json(progress, {"stage": "complete", "file": target.name})
    return result
