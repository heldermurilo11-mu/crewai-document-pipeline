"""Offline tests for SQLite WAL Checkpoint, InstanceLock, RuntimeConfig, and DocumentPipeline."""

from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from runtime.checkpoint import Checkpoint, PublicationConflict
from runtime.config import RuntimeConfig
from runtime.deadline import Deadline, DeadlineExceeded
from runtime.files import AlreadyRunning, InstanceLock
from runtime.pipeline import DocumentPipeline
from runtime.supervisor import WorkerFailure
from tools.reviewer import Reviewer


class FakeDrive:
    def __init__(self, sources):
        self.sources = list(sources)
        self.moves = []
        self.move_ok = True
        self.credentials_dict = None
        self.token_dict = None

    def list_files(self, _folder_id):
        return list(self.sources)

    def get_or_create_subfolder(self, parent_id, name):
        return f"{parent_id}:{name}"

    def move_to_folder(self, file_id, destination_folder_id, file_name=""):
        self.moves.append((file_id, destination_folder_id, file_name))
        return self.move_ok


def make_extraction(code="1823875", ident="PONTO NORTE", date="10/02/2026"):
    return {
        "template": "LAB_ALPHA",
        "file": "report.pdf",
        "source_sha256": f"sha-{code}",
        "covered_pages": [1],
        "match_method": "anchor",
        "confidence": 1.0,
        "samples": [
            {
                "template": "LAB_ALPHA",
                "codigo": code,
                "identificacao": ident,
                "data": date,
                "hora": "09:45",
                "source_pages": [1],
            }
        ],
    }


def make_payload(code="1823875", ident="PONTO NORTE"):
    return {
        "arquivo": "report.pdf",
        "codigo": code,
        "identificacao": ident,
        "data_hora": "10/02/2026 09:45",
        "template": "LAB_ALPHA",
        "status": "APROVADO",
        "motivo": "",
    }


class CheckpointAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)
        self.store = Checkpoint(self.state)
        self.doc = self.store.discover("drive-1", "rev-1", "folder", "report.pdf", "csv")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_recovery_marks_inflight_uncertain_and_extracting_deferred(self):
        self.store.stage(self.doc["id"], make_extraction(), [make_payload()])
        self.store.update_sample("1823875", "SENDING")
        other = self.store.discover("drive-2", "rev-1", "folder", "second.pdf", "csv")
        self.store.update_document(other["id"], "EXTRACTING")
        self.store.close()

        self.store = Checkpoint(self.state)
        recovered = self.store.recover_inflight()
        self.assertEqual(recovered, 2)
        self.assertEqual(self.store.sample("1823875")["status"], "UNCERTAIN")
        self.assertEqual(self.store.document(other["id"])["status"], "DEFERRED")

    def test_stage_rejects_conflicting_sample_attributes(self):
        self.store.stage(self.doc["id"], make_extraction(), [make_payload()])
        other = self.store.discover("drive-2", "rev-1", "folder", "second.pdf", "csv")
        with self.assertRaises(PublicationConflict):
            self.store.stage(
                other["id"],
                make_extraction(),
                [make_payload(ident="CONFLICTING ID")],
            )

    def test_stage_supports_identification_only_samples_without_codigo(self):
        payload = make_payload(code="", ident="PONTO-SEM-CODIGO")
        self.store.stage(self.doc["id"], make_extraction(code=""), [payload])
        rows = self.store.samples(self.doc["id"])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["code"].startswith("anon_"))

    def test_backup_preserves_committed_state(self):
        self.store.stage(self.doc["id"], make_extraction(), [make_payload()])
        backup_path = self.state / "backup.sqlite3"
        self.store.backup(backup_path)
        with closing(sqlite3.connect(backup_path)) as conn:
            row = conn.execute("SELECT status FROM documents WHERE id=?", (self.doc["id"],)).fetchone()
            self.assertEqual(row[0], "VALIDATED")

    def test_instance_lock_excludes_overlap(self):
        with InstanceLock(self.state / "pipeline.lock"):
            with self.assertRaises(AlreadyRunning):
                with InstanceLock(self.state / "pipeline.lock"):
                    self.fail("Overlapping run acquired the lock")
        with InstanceLock(self.state / "pipeline.lock"):
            pass

    def test_invalid_budgets_fail_validation(self):
        with self.assertRaises(ValueError):
            RuntimeConfig(batch_budget_seconds=100, file_budget_seconds=100).validate()
        with self.assertRaises(ValueError):
            RuntimeConfig(retry_cooldown_seconds=-1).validate()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Checkpoint(self.root / "state")
        self.drive = FakeDrive([{"id": "src-1", "name": "report.pdf", "md5Checksum": "rev-1"}])
        self.calls = 0
        self.config = replace(
            RuntimeConfig(),
            state_dir=str(self.root / "state"),
            output_csv=str(self.root / "output" / "resultado_crew.csv"),
            dry_run=False,
            orientation_enabled=False,
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _run(self, *, config=None, worker=None, deadline=None):
        def default_worker(*_a, **_k):
            self.calls += 1
            return make_extraction()

        pipeline = DocumentPipeline(
            config or self.config,
            deadline or Deadline(1500),
            self.store,
            self.drive,
            Reviewer(),
            api_key="fake",
            worker=worker or default_worker,
        )
        return pipeline.run("folder-1")

    def test_dry_run_exports_csv_without_drive_moves(self):
        result = self._run(config=replace(self.config, dry_run=True))
        self.assertEqual(result["files_processed"], 1)
        self.assertEqual(result["samples_exported"], 1)
        self.assertEqual(self.drive.moves, [])
        csv_text = Path(result["csv_path"]).read_text(encoding="utf-8")
        self.assertIn("1823875;PONTO NORTE;10/02/2026 09:45;LAB_ALPHA;APROVADO", csv_text)

    def test_failed_drive_move_retries_only_movement(self):
        self.drive.move_ok = False
        first = self._run()
        self.assertEqual(first["files_failed"], 1)
        self.assertEqual(self.store.all_documents()[0]["status"], "FINALIZATION_PENDING")

        self.drive.move_ok = True
        second = self._run()
        self.assertEqual(second["files_processed"], 1)
        self.assertEqual(self.calls, 1)  # Never re-ran expensive OCR worker
        self.assertEqual(self.store.all_documents()[0]["status"], "DONE")

    def test_invalid_sample_blocks_document_in_review_required(self):
        bad = make_extraction(code="12345")  # Not 7 digits
        result = self._run(worker=lambda *_a, **_k: bad)
        self.assertEqual(result["files_review_required"], 1)
        self.assertEqual(self.drive.moves, [])
        doc = self.store.all_documents()[0]
        self.assertEqual(doc["status"], "REVIEW_REQUIRED")

    def test_insufficient_batch_budget_defers_without_worker(self):
        result = self._run(deadline=Deadline(30))
        self.assertEqual(self.calls, 0)
        self.assertEqual(result["files_deferred"], 1)

    def test_worker_timeout_defers_with_cooldown(self):
        def timeout_worker(*_a, **_k):
            raise DeadlineExceeded("controlled timeout")

        result = self._run(worker=timeout_worker)
        doc = self.store.all_documents()[0]
        self.assertEqual(doc["status"], "DEFERRED")
        self.assertGreater(doc["next_attempt"], 0)
        self.assertEqual(result["files_failed"], 1)

    def test_worker_failure_trips_circuit_breaker(self):
        self.drive.sources = [
            {"id": f"src-{i}", "name": f"doc-{i}.pdf", "md5Checksum": f"m-{i}"}
            for i in range(3)
        ]

        def failing_worker(*_a, **_k):
            self.calls += 1
            raise WorkerFailure("API quota unavailable")

        result = self._run(worker=failing_worker)
        self.assertEqual(self.calls, 1)
        self.assertIsNotNone(result["circuit_breaker"])
        self.assertEqual(result["files_failed"], 1)

    def test_ten_documents_preserve_distinct_identities_and_csv_rows(self):
        self.drive.sources = [
            {"id": f"d{i}", "name": f"doc-{i}.pdf", "md5Checksum": f"r{i}"}
            for i in range(10)
        ]

        def distinct_worker(_target, arguments, *_a, **_k):
            code = "18" + str(int(arguments["download"]["id"][1:])).zfill(5)
            return make_extraction(code=code, ident=f"POINT-{code}")

        result = self._run(worker=distinct_worker)
        self.assertEqual(result["files_processed"], 10)
        self.assertEqual(result["samples_exported"], 10)
        self.assertEqual(len(self.drive.moves), 10)
        lines = Path(result["csv_path"]).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 11)  # header + 10 records


if __name__ == "__main__":
    unittest.main()
