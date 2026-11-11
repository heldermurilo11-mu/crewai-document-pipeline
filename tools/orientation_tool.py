"""Per-page visual orientation detector using Gemini Vision."""

import json
import logging
import os
from pathlib import Path
from typing import Iterable

from google.genai import types

from runtime.deadline import Deadline
from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

ORIENTATION_SCHEMA = {
    "type": "OBJECT",
    "required": ["rotations"],
    "properties": {
        "rotations": {
            "type": "ARRAY",
            "items": {
                "type": "INTEGER",
                "enum": [0, 90, 180, 270],
            },
        }
    },
}


class OrientationTool:
    """Detects clockwise clockwise correction angles `[0, 90, 180, 270]` per page."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL_FAST", "gemini-2.5-flash")

    def detect_images(
        self,
        client,
        image_paths: Iterable[str | Path],
        *,
        deadline: Deadline | None = None,
        request_timeout_ms: int = 120_000,
    ) -> list[int]:
        """Return clockwise clockwise rotation angles needed to make each page upright."""
        paths = [Path(p) for p in image_paths]
        if not paths:
            return []

        prompt = (
            f"Analyze these {len(paths)} document page image(s) in order (page 1 to page {len(paths)}). "
            "For EACH page, return the clockwise rotation in degrees needed so printed text and tables "
            "read upright left-to-right: 0 if upright, 90 if rotated 90 degrees counter-clockwise, "
            "180 if upside-down, 270 if rotated 90 degrees clockwise. "
            f'Return JSON {{"rotations": [<int>, ...]}} with exactly {len(paths)} entries.'
        )

        contents = []
        for idx, img_path in enumerate(paths, 1):
            contents.append(f"Page {idx}:")
            contents.append(
                types.Part.from_bytes(
                    data=img_path.read_bytes(),
                    mime_type="image/png",
                )
            )
        contents.append(prompt)

        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": ORIENTATION_SCHEMA,
            "temperature": 0.0,
        }
        if deadline is not None:
            config_kwargs["http_options"] = types.HttpOptions(
                timeout=deadline.timeout_ms(request_timeout_ms),
                retry_options=types.HttpRetryOptions(attempts=1),
            )

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        raw_text = (getattr(response, "text", None) or "").strip()
        if not raw_text:
            return [0] * len(paths)
        parsed = json.loads(raw_text)
        rotations = parsed.get("rotations") if isinstance(parsed, dict) else parsed
        if not isinstance(rotations, list):
            return [0] * len(paths)
        normalized = []
        for idx in range(len(paths)):
            angle = int(rotations[idx]) if idx < len(rotations) else 0
            normalized.append(angle if angle in (0, 90, 180, 270) else 0)
        return normalized

    def correct_pdf(
        self,
        pdf_bytes: bytes,
        rotations: list[int],
    ) -> bytes:
        """Apply per-page clockwise rotation corrections to PDF bytes in-memory."""
        updated = pdf_bytes
        for idx, angle in enumerate(rotations):
            if angle in (90, 180, 270):
                updated = PDFProcessor.rotate_pdf(updated, angle, page_index=idx)
        return updated
