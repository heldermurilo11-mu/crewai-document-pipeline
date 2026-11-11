"""Validated runtime configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable wall-clock, retry, extraction, and storage configuration."""

    state_dir: str = "state"
    output_csv: str = "output/resultado_crew.csv"
    batch_budget_seconds: float = 1500.0
    file_budget_seconds: float = 600.0
    worker_kill_grace_seconds: float = 5.0
    finalize_reserve_seconds: float = 60.0
    supervisor_grace_seconds: float = 60.0
    termination_grace_seconds: float = 10.0
    retry_cooldown_seconds: float = 300.0
    gemini_request_timeout_ms: int = 900_000
    extraction_max_attempts: int = 2
    max_document_attempts: int = 3
    drive_request_timeout_seconds: float = 120.0
    orientation_enabled: bool = True
    streaming_enabled: bool = True
    gemini_model_pro: str = "gemini-2.5-pro"
    gemini_model_fast: str = "gemini-2.5-flash"
    dry_run: bool = False
    move_drive_files: bool = True

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        config = cls(
            state_dir=_env_str("PIPELINE_STATE_DIR", "state"),
            output_csv=_env_str("PIPELINE_OUTPUT_CSV", "output/resultado_crew.csv"),
            batch_budget_seconds=_env_float("BATCH_BUDGET_SECONDS", 1500.0),
            file_budget_seconds=_env_float("FILE_BUDGET_SECONDS", 600.0),
            worker_kill_grace_seconds=_env_float("WORKER_KILL_GRACE_SECONDS", 5.0),
            finalize_reserve_seconds=_env_float("FINALIZE_RESERVE_SECONDS", 60.0),
            supervisor_grace_seconds=_env_float("SUPERVISOR_GRACE_SECONDS", 60.0),
            termination_grace_seconds=_env_float("TERMINATION_GRACE_SECONDS", 10.0),
            retry_cooldown_seconds=_env_float("RETRY_COOLDOWN_SECONDS", 300.0),
            gemini_request_timeout_ms=_env_int("GEMINI_REQUEST_TIMEOUT_MS", 900_000),
            extraction_max_attempts=_env_int("EXTRACTION_MAX_ATTEMPTS", 2),
            max_document_attempts=_env_int("MAX_DOCUMENT_ATTEMPTS", 3),
            drive_request_timeout_seconds=_env_float("DRIVE_REQUEST_TIMEOUT_SECONDS", 120.0),
            orientation_enabled=_env_bool("ORIENTATION_ENABLED", True),
            streaming_enabled=_env_bool("STREAMING_ENABLED", True),
            gemini_model_pro=_env_str("GEMINI_MODEL_PRO", "gemini-2.5-pro"),
            gemini_model_fast=_env_str("GEMINI_MODEL_FAST", "gemini-2.5-flash"),
            dry_run=_env_bool("DRY_RUN", False),
            move_drive_files=_env_bool("MOVE_DRIVE_FILES", True),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.batch_budget_seconds <= 0 or self.file_budget_seconds <= 0:
            raise ValueError("Budget values must be positive")
        if min(
            self.worker_kill_grace_seconds,
            self.finalize_reserve_seconds,
            self.supervisor_grace_seconds,
            self.termination_grace_seconds,
            self.retry_cooldown_seconds,
        ) < 0:
            raise ValueError("Reserve, grace, and cooldown values cannot be negative")
        if (
            self.file_budget_seconds
            + self.worker_kill_grace_seconds
            + self.finalize_reserve_seconds
            >= self.batch_budget_seconds
        ):
            raise ValueError(
                "file_budget_seconds + worker_kill_grace_seconds + finalize_reserve_seconds "
                "must stay strictly below batch_budget_seconds"
            )
        if self.gemini_request_timeout_ms < 1000:
            raise ValueError("gemini_request_timeout_ms must be at least 1000 ms")
        if self.drive_request_timeout_seconds <= 0:
            raise ValueError("drive_request_timeout_seconds must be positive")
        if self.extraction_max_attempts < 1 or self.max_document_attempts < 1:
            raise ValueError("Attempt limits must be >= 1")

    def resolved_state_dir(self) -> Path:
        return Path(self.state_dir).expanduser().resolve()

    def resolved_output_csv(self) -> Path:
        return Path(self.output_csv).expanduser().resolve()
