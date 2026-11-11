"""Resilient checkpoint-backed batch orchestrator for document extraction and export."""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Optional

from .checkpoint import Checkpoint, PublicationConflict
from .config import RuntimeConfig
from .deadline import Deadline, DeadlineExceeded
from .files import atomic_write_text
from .supervisor import WorkerFailure, run_worker

logger = logging.getLogger(__name__)


def _format_csv_rows(rows: list[dict]) -> str:
    lines = ["Arquivo;Codigo;Identificacao;DataHora;Template;Status;Motivo"]
    for row in rows:
        arquivo = str(row.get("arquivo", "")).replace(";", ",")
        codigo = str(row.get("codigo", "")).replace(";", ",")
        ident = str(row.get("identificacao", "")).replace(";", ",")
        data_hora = str(row.get("data_hora", "")).replace(";", ",")
        template = str(row.get("template", "GENERIC")).replace(";", ",")
        status = str(row.get("status", "")).replace(";", ",")
        motivo = str(row.get("motivo", "")).replace(";", ",")
        lines.append(f"{arquivo};{codigo};{ident};{data_hora};{template};{status};{motivo}")
    return "\n".join(lines) + "\n"


def _sample_record(sample: dict, file_name: str, approved: bool, issues: list[str]) -> dict:
    data_str = str(sample.get("data", "") or "").strip()
    hora_str = str(sample.get("hora", "") or "").strip()
    data_hora = f"{data_str} {hora_str}".strip()
    return {
        "arquivo": file_name,
        "codigo": str(sample.get("codigo", "") or "").strip(),
        "identificacao": str(sample.get("identificacao", "") or "").strip(),
        "data_hora": data_hora,
        "template": str(sample.get("template", "GENERIC") or "GENERIC").strip(),
        "status": "APROVADO" if approved else "REJEITADO",
        "motivo": ", ".join(issues) if not approved else "",
    }


class DocumentPipeline:
    """Crash-safe document pipeline integrating discovery, worker supervision, review, and export."""

    def __init__(
        self,
        config: RuntimeConfig,
        deadline: Deadline,
        store: Checkpoint,
        drive: Any,
        reviewer: Any,
        *,
        api_key: str = "",
        feedback_store: Any = None,
        worker: Callable = run_worker,
        on_document: Optional[Callable[[dict, dict], None]] = None,
    ):
        self.config = config
        self.deadline = deadline
        self.store = store
        self.drive = drive
        self.reviewer = reviewer
        self.api_key = api_key
        self.feedback_store = feedback_store
        self.worker = worker
        self.on_document = on_document or (lambda _doc, _summary: None)

    def _export_consolidated_csv(self) -> Path:
        rows: list[dict] = []
        for doc in self.store.all_documents():
            if doc["status"] in {"DONE", "VALIDATED", "FINALIZATION_PENDING"}:
                for sample_row in self.store.samples(doc["id"]):
                    payload = json.loads(sample_row["payload"])
                    rows.append(payload)
            elif doc["status"] == "REVIEW_REQUIRED":
                rows.append(
                    {
                        "arquivo": doc["name"],
                        "codigo": "",
                        "identificacao": "",
                        "data_hora": "",
                        "template": "REVIEW_REQUIRED",
                        "status": "REJEITADO",
                        "motivo": doc["error"] or "Rejeitado na revisão automática",
                    }
                )
        csv_path = self.config.resolved_output_csv()
        atomic_write_text(csv_path, _format_csv_rows(rows))
        return csv_path

    def run(self, folder_id: str, target: str = "csv", *, local_sources: list[dict] | None = None) -> dict:
        """Process documents from Google Drive *folder_id* (or *local_sources*) safely."""
        self.store.recover_inflight()
        drive_read_only = (not self.config.move_drive_files) or (local_sources is not None)
        sources = (
            local_sources
            if local_sources is not None
            else self.drive.list_files(folder_id)
        )
        source_by_id: dict[str, dict] = {}
        for item in sources:
            revision = item.get("md5Checksum") or item.get("modifiedTime") or item.get("sha256") or "v1"
            doc = self.store.discover(
                source_id=item["id"],
                revision=revision,
                folder_id=folder_id,
                name=item["name"],
                target=target,
            )
            source_by_id[doc["id"]] = item

        summary: dict[str, Any] = {
            "status": "success",
            "files_discovered": len(sources),
            "files_processed": 0,
            "files_deferred": 0,
            "files_failed": 0,
            "files_review_required": 0,
            "samples_exported": 0,
            "samples_rejected": 0,
            "rejected_files": [],
            "dry_run": self.config.dry_run,
            "csv_path": None,
            "circuit_breaker": None,
        }

        documents = self.store.pending(folder_id)
        now = time.time()

        for document in documents:
            identity = document["id"]
            name = document["name"]
            source = source_by_id.get(identity)
            if not source and document["status"] != "FINALIZATION_PENDING":
                continue

            if document["next_attempt"] and document["next_attempt"] > now and not document["extraction"]:
                summary["files_deferred"] += 1
                continue

            if document["attempts"] >= self.config.max_document_attempts and not document["extraction"]:
                self.store.update_document(
                    identity,
                    "REVIEW_REQUIRED",
                    f"Maximum attempts ({self.config.max_document_attempts}) exhausted",
                )
                summary["rejected_files"].append(name)
                continue

            started = time.monotonic()
            self.store.event(identity, "document_start", status=document["status"])

            try:
                if not document["extraction"]:
                    try:
                        budget = self.deadline.timeout(
                            self.config.file_budget_seconds,
                            reserve=self.config.worker_kill_grace_seconds
                            + self.config.finalize_reserve_seconds,
                        )
                    except DeadlineExceeded:
                        summary["files_deferred"] += 1
                        self.store.event(identity, "deferred", reason="insufficient_batch_budget")
                        continue

                    workspace = self.store.directory / "work" / identity
                    workspace.mkdir(parents=True, exist_ok=True)
                    local_path = (
                        Path(source["local_path"])
                        if source and source.get("local_path")
                        else workspace / Path(name).name
                    )

                    self.store.update_document(identity, "EXTRACTING", attempts=document["attempts"] + 1)
                    self.store.event(identity, "worker_start", budget_seconds=budget)

                    download_spec = None
                    if source and not source.get("local_path"):
                        download_spec = {
                            "id": document["source_id"],
                            "name": name,
                            "credentials": getattr(self.drive, "credentials_dict", None),
                            "token": getattr(self.drive, "token_dict", None),
                        }

                    extraction = self.worker(
                        "runtime.ocr_worker:extract",
                        {
                            "file_path": str(local_path),
                            "api_key": self.api_key,
                            "config_values": asdict(self.config),
                            "seconds": budget,
                            "progress_path": str(workspace / "progress.json"),
                            "expected_md5": source.get("md5Checksum") if source else None,
                            "download": download_spec,
                        },
                        budget,
                        workspace / "worker",
                        grace=self.config.termination_grace_seconds,
                    )

                    review = self.reviewer.review_extraction(json.dumps(extraction, ensure_ascii=False))
                    if not review.get("all_approved"):
                        summary["samples_rejected"] += review.get("rejected_count", 0)
                        reason = review.get("rejection_summary") or "Extraction rejected"
                        self.store.update_document(identity, "REVIEW_REQUIRED", reason)
                        self.store.event(identity, "review_rejected", rejected_count=review.get("rejected_count", 0))
                        if self.feedback_store is not None:
                            self.feedback_store.record_failure(
                                str(local_path),
                                review.get("template", "GENERIC"),
                                {},
                                extraction.get("samples", []),
                                reason,
                            )
                        summary["files_failed"] += 1
                        summary["rejected_files"].append(name)
                        continue

                    samples = [item["sample"] for item in review["results"]]
                    extraction["samples"] = samples
                    payloads = [_sample_record(sample, name, True, []) for sample in samples]
                    self.store.stage(
                        identity,
                        extraction,
                        payloads,
                        extraction.get("source_sha256", ""),
                    )
                    if self.feedback_store is not None:
                        self.feedback_store.record_success(
                            str(local_path),
                            extraction.get("template", "GENERIC"),
                            {},
                            samples,
                            extraction.get("match_method", "unknown"),
                            float(extraction.get("confidence", 0.0) or 0.0),
                        )

                # Mark samples confirmed against durable CSV output
                csv_path = self._export_consolidated_csv()
                summary["csv_path"] = str(csv_path)
                staged_samples = self.store.samples(identity)
                for sample_row in staged_samples:
                    if sample_row["status"] != "CONFIRMED":
                        self.store.update_sample(sample_row["code"], "SENDING")
                        self.store.update_sample(sample_row["code"], "CONFIRMED", remote_id=csv_path.name)
                summary["samples_exported"] += len(staged_samples)

                if self.config.dry_run or drive_read_only:
                    self.store.update_document(
                        identity,
                        "DONE" if local_sources is not None else "FINALIZATION_PENDING",
                        "Dry-run or Drive move disabled" if not local_sources else "",
                    )
                    summary["files_processed"] += 1
                    continue

                self.deadline.check("drive finalization", reserve=self.config.finalize_reserve_seconds)
                dest_folder = self.drive.get_or_create_subfolder(folder_id, "processados")
                if not self.drive.move_to_folder(document["source_id"], dest_folder, name):
                    self.store.update_document(
                        identity, "FINALIZATION_PENDING", "Drive did not confirm movement"
                    )
                    summary["files_failed"] += 1
                    continue
                self.store.update_document(identity, "DONE")
                summary["files_processed"] += 1

            except DeadlineExceeded:
                latest = self.store.document(identity)
                progress = self.store.directory / "work" / identity / "progress.json"
                phase = (
                    json.loads(progress.read_text(encoding="utf-8")).get("stage", "worker")
                    if progress.exists()
                    else "worker"
                )
                if latest and latest["extraction"]:
                    summary["files_deferred"] += 1
                else:
                    self.store.update_document(
                        identity,
                        "DEFERRED",
                        f"File deadline exceeded during {phase}",
                        next_attempt=time.time() + self.config.retry_cooldown_seconds,
                    )
                    summary["files_failed"] += 1
                self.store.event(identity, "deadline", phase=phase)
            except PublicationConflict as exc:
                self.store.update_document(identity, "REVIEW_REQUIRED", str(exc))
                summary["files_failed"] += 1
                summary["rejected_files"].append(name)
            except WorkerFailure as exc:
                self.store.update_document(
                    identity,
                    "DEFERRED",
                    str(exc),
                    next_attempt=time.time() + self.config.retry_cooldown_seconds,
                )
                summary["files_failed"] += 1
                summary["circuit_breaker"] = str(exc)
                break
            except Exception as exc:
                latest = self.store.document(identity)
                next_status = "FINALIZATION_PENDING" if latest and latest["extraction"] else "DEFERRED"
                self.store.update_document(
                    identity,
                    next_status,
                    f"{type(exc).__name__}: {exc}",
                    next_attempt=time.time() + self.config.retry_cooldown_seconds,
                )
                summary["files_failed"] += 1
                summary["circuit_breaker"] = type(exc).__name__
                self.store.event(identity, "error", error_type=type(exc).__name__)
                break
            finally:
                current_doc = self.store.document(identity)
                if current_doc:
                    self.store.event(
                        identity,
                        "document_end",
                        duration_seconds=time.monotonic() - started,
                        state=current_doc["status"],
                    )
                    self.on_document(current_doc, summary)

        # Refresh final consolidated CSV when at least one processed/rejected item exists
        if self.store.all_documents():
            summary["csv_path"] = str(self._export_consolidated_csv())

        summary["files_review_required"] = self.store.db.execute(
            "SELECT count(*) FROM documents WHERE folder_id=? AND status='REVIEW_REQUIRED'",
            (folder_id,),
        ).fetchone()[0]
        if summary["files_failed"] or summary["files_review_required"]:
            summary["status"] = "partial" if summary["files_processed"] > 0 else "failed"
        elif not sources and not documents:
            summary["status"] = "no_files"
        return summary
