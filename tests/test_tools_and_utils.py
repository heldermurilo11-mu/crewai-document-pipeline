"""Offline tests for PatternIdentifier, Reviewer, PDFProcessor, utils, and CrewAI tool adapters."""

from pathlib import Path
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
from pypdf import PdfReader, PdfWriter

import tools.crew_wrappers as crew_wrappers
from tools.feedback_store import FeedbackStore
from tools.filler_agent import FillerAgentTool
from tools.pattern_identifier import PatternIdentifierTool
from tools.pdf_processor import PDFProcessor
from tools.reviewer import Reviewer
from tools.utils import normalize_code, normalize_time


def make_text_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((36, 72), text)
    doc.save(str(path))
    doc.close()


class ToolsAndWrappersTests(unittest.TestCase):
    def test_normalize_code_and_time(self):
        self.assertEqual(normalize_code("01823875", known_prefixes=["18", "19"]), "1823875")
        self.assertEqual(normalize_code("11823875", known_prefixes=["18", "19"]), "1823875")
        self.assertEqual(normalize_code("1823875"), "1823875")
        self.assertEqual(normalize_time("10h35"), "10:35")
        self.assertEqual(normalize_time("09:05"), "09:05")

    def test_reviewer_validates_codes_dates_and_identifications(self):
        reviewer = Reviewer()
        ok = reviewer.review_extraction(
            json.dumps(
                {
                    "template": "LAB_ALPHA",
                    "samples": [
                        {
                            "codigo": "0001234",
                            "identificacao": "PONTO NORTE",
                            "data": "10/02/26",
                            "hora": "09:45",
                        }
                    ],
                }
            )
        )
        self.assertTrue(ok["all_approved"])
        self.assertEqual(ok["results"][0]["sample"]["data"], "10/02/2026")

        bad = reviewer.review_extraction(
            json.dumps(
                {
                    "template": "LAB_ALPHA",
                    "samples": [{"codigo": "123", "identificacao": "X", "data": "31/02/2026"}],
                }
            )
        )
        self.assertFalse(bad["all_approved"])
        self.assertEqual(bad["rejected_count"], 1)

    def test_pattern_identifier_matches_lab_alpha_and_lab_beta_via_pymupdf(self):
        with tempfile.TemporaryDirectory() as directory:
            alpha = Path(directory) / "alpha.pdf"
            make_text_pdf(alpha, "CADEIA DE CUSTODIA - LAB ALPHA\nLAB ALPHA ANALISES")
            res = PatternIdentifierTool(api_key="").identify(str(alpha), allow_inference=False)
            self.assertEqual(res["template_name"], "LAB_ALPHA")
            self.assertEqual(res["match_method"], "anchor")

            beta = Path(directory) / "beta.pdf"
            make_text_pdf(beta, "FORMULARIO DE CUSTODIA - LAB BETA\nLAB BETA AMBIENTAL")
            res_b = PatternIdentifierTool(api_key="").identify(str(beta), allow_inference=False)
            self.assertEqual(res_b["template_name"], "LAB_BETA")
            self.assertEqual(res_b["match_method"], "anchor")

    def test_pdf_processor_rotates_in_memory(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=100)
        buf = io.BytesIO()
        writer.write(buf)
        rotated = PDFProcessor.rotate_pdf(buf.getvalue(), 90)
        reader = PdfReader(io.BytesIO(rotated))
        self.assertEqual(reader.pages[0].get("/Rotate"), 90)

    def test_crew_wrappers_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            make_text_pdf(pdf_path, "LABORATORIO MODELO ALPHA\nCADEIA DE CUSTODIA\nPONTO DE IDENTIFICACAO")
            csv_out = root / "output" / "resultado_crew.csv"
            state_dir = root / "state"
            fb_dir = root / "feedback"

            fake_extraction = {
                "template": "LAB_ALPHA",
                "file": "sample.pdf",
                "source_sha256": "abc123",
                "match_method": "anchor",
                "confidence": 1.0,
                "samples": [
                    {
                        "template": "LAB_ALPHA",
                        "codigo": "0001234",
                        "identificacao": "PONTO A1",
                        "data": "10/02/2026",
                        "hora": "08:30",
                        "source_pages": [1],
                    }
                ],
            }

            with patch.dict(
                os.environ,
                {
                    "PIPELINE_STATE_DIR": str(state_dir),
                    "PIPELINE_OUTPUT_CSV": str(csv_out),
                    "DRY_RUN": "true",
                    "MOVE_DRIVE_FILES": "false",
                },
                clear=False,
            ), patch.object(
                FillerAgentTool, "extract", return_value=fake_extraction
            ), patch.object(
                crew_wrappers, "feedback_tool", FeedbackStore(feedback_dir=fb_dir)
            ):
                extracted_json = crew_wrappers.extrair_dados_cadeia_custodia(
                    json.dumps(
                        {
                            "arquivos": [
                                {"file_id": "fid-1", "file_name": "sample.pdf", "file_path": str(pdf_path)}
                            ]
                        }
                    )
                )
                summary = crew_wrappers.validar_e_exportar_dados(extracted_json)

            self.assertIn("Aprovadas: 1, Rejeitadas: 0", summary)
            self.assertTrue(csv_out.is_file())
            self.assertIn("0001234;PONTO A1;10/02/2026 08:30;LAB_ALPHA;APROVADO", csv_out.read_text(encoding="utf-8"))
            self.assertTrue(pdf_path.exists(), "Dry-run must preserve source files")


if __name__ == "__main__":
    unittest.main()
