"""Unified CLI entrypoint supporting CrewAI orchestration, resilient runtime mode, and watchdog supervision."""

import argparse
from dataclasses import replace
import hashlib
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from crew import ProtocrewCrew
from runtime.checkpoint import Checkpoint
from runtime.config import RuntimeConfig
from runtime.deadline import Deadline
from runtime.files import InstanceLock
from runtime.pipeline import DocumentPipeline
from runtime.supervisor import supervise_command, wait_start_gate
from tools._crew_compat import CREWAI_AVAILABLE
from tools.drive_tools import GoogleDriveDownloadTool
from tools.feedback_store import FeedbackStore
from tools.reviewer import Reviewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _discover_local_pdfs(local_dir: Path) -> list[dict]:
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local input directory not found: {local_dir}")
    sources = []
    for path in sorted(local_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"}:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            sources.append(
                {
                    "id": f"local_{hashlib.sha256(str(path.resolve()).encode('utf-8')).hexdigest()[:16]}",
                    "name": path.name,
                    "md5Checksum": digest,
                    "local_path": str(path.resolve()),
                }
            )
    return sources


def run_resilient_pipeline(config: RuntimeConfig, folder_id: str, *, local_dir: Path | None = None) -> dict:
    deadline = Deadline(config.batch_budget_seconds)
    store = Checkpoint(config.resolved_state_dir())
    try:
        drive = None if local_dir else GoogleDriveDownloadTool(
            deadline=deadline,
            request_timeout=config.drive_request_timeout_seconds,
        )
        local_sources = _discover_local_pdfs(local_dir) if local_dir else None
        pipeline = DocumentPipeline(
            config=config,
            deadline=deadline,
            store=store,
            drive=drive,
            reviewer=Reviewer(),
            api_key=os.getenv("GEMINI_API_KEY", ""),
            feedback_store=FeedbackStore(),
        )
        return pipeline.run(folder_id or "local", local_sources=local_sources)
    finally:
        store.close()


def run(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="CrewAI Document Pipeline (PROTOCREW + Production Runtime)")
    parser.add_argument(
        "--mode",
        choices=("crew", "pipeline", "supervise"),
        default=os.getenv("PIPELINE_MODE", "crew").strip().lower() or "crew",
        help="Execution mode: CrewAI agents ('crew'), resilient checkpoint pipeline ('pipeline'), or watchdog ('supervise').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Disable remote Drive mutations.")
    parser.add_argument("--no-move", action="store_true", help="Keep source files in original Drive folder.")
    parser.add_argument("--local-dir", type=Path, default=None, help="Process local PDFs/images instead of Google Drive.")
    args = parser.parse_args(argv)

    config = RuntimeConfig.from_env()
    if args.dry_run:
        config = replace(config, dry_run=True)
        os.environ["DRY_RUN"] = "true"
    if args.no_move:
        config = replace(config, move_drive_files=False)
        os.environ["MOVE_DRIVE_FILES"] = "false"

    if args.mode == "supervise":
        cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "pipeline"]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.no_move:
            cmd.append("--no-move")
        if args.local_dir:
            cmd.extend(["--local-dir", str(args.local_dir)])
        return supervise_command(cmd, config)

    wait_start_gate()
    os.environ["MODEL"] = f"gemini/{config.gemini_model_pro}"

    folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
    if args.mode == "crew":
        if not CREWAI_AVAILABLE:
            raise RuntimeError(
                "Pacote 'crewai' não instalado. Instale via `pip install -r requirements.txt` "
                "ou execute com `--mode pipeline`."
            )
        if not folder_id:
            raise ValueError(
                "Variável de ambiente DRIVE_FOLDER_ID não configurada. "
                "Copie .env.example para .env e informe o ID da pasta do Google Drive."
            )
        with InstanceLock(config.resolved_state_dir() / "pipeline.lock"):
            logger.info("Iniciando ProtocrewCrew (CrewAI sequencial + ferramentas hardened)...")
            ProtocrewCrew().crew().kickoff(inputs={"folder_id": folder_id})
            return 0

    if not args.local_dir and not folder_id:
        raise ValueError("Configure DRIVE_FOLDER_ID no .env ou informe --local-dir.")

    with InstanceLock(config.resolved_state_dir() / "pipeline.lock"):
        summary = run_resilient_pipeline(config, folder_id, local_dir=args.local_dir)
        logger.info("Pipeline finalizado: %s", summary)
        return 0 if summary["status"] in {"success", "no_files"} else 1


if __name__ == "__main__":
    raise SystemExit(run())
