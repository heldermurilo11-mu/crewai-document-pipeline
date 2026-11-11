"""Durable SQLite WAL checkpoint journal for crash-safe document pipelines."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time


class PublicationConflict(RuntimeError):
    """Raised when deterministic sample keys carry conflicting attributes."""


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def document_identity(source_id: str, revision: str) -> str:
    digest = hashlib.sha256(f"{source_id}:{revision}".encode("utf-8")).hexdigest()[:24]
    return f"doc_{digest}"


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt REAL NOT NULL DEFAULT 0,
    content_sha256 TEXT NOT NULL DEFAULT '',
    extraction TEXT,
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    code TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    target TEXT NOT NULL,
    status TEXT NOT NULL,
    remote_id TEXT,
    error TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS document_samples (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    code TEXT NOT NULL REFERENCES samples(code) ON DELETE CASCADE,
    PRIMARY KEY (document_id, code)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    stage TEXT NOT NULL,
    created_at REAL NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    task_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    row_data TEXT NOT NULL,
    sent INTEGER NOT NULL DEFAULT 0
);
"""


class Checkpoint:
    """Crash-safe SQLite WAL state store for discovery, extraction, review, and export."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "journal.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def recover_inflight(self) -> int:
        """Downgrade ambiguous in-flight states after an abrupt process kill."""
        now = time.time()
        with self.transaction():
            sample_cur = self.db.execute(
                "UPDATE samples SET status='UNCERTAIN', "
                "error='Recovered in-flight export state; manual confirmation required', "
                "updated_at=? WHERE status='SENDING'",
                (now,),
            )
            doc_cur = self.db.execute(
                "UPDATE documents SET status='DEFERRED', "
                "error='Recovered interrupted extraction worker', updated_at=? "
                "WHERE status='EXTRACTING'",
                (now,),
            )
            return sample_cur.rowcount + doc_cur.rowcount

    def discover(
        self,
        source_id: str,
        revision: str,
        folder_id: str,
        name: str,
        target: str = "csv",
    ) -> dict:
        identity = document_identity(source_id, revision)
        now = time.time()
        with self.transaction():
            existing = self.db.execute(
                "SELECT * FROM documents WHERE id=?",
                (identity,),
            ).fetchone()
            if existing:
                return dict(existing)
            self.db.execute(
                "INSERT INTO documents(id,source_id,revision,folder_id,name,target,status,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,'DISCOVERED',?,?)",
                (identity, source_id, revision, folder_id, name, target, now, now),
            )
            row = self.db.execute("SELECT * FROM documents WHERE id=?", (identity,)).fetchone()
            return dict(row)

    def document(self, identity: str) -> dict | None:
        row = self.db.execute("SELECT * FROM documents WHERE id=?", (identity,)).fetchone()
        return dict(row) if row else None

    def pending(self, folder_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM documents WHERE folder_id=? "
            "AND status NOT IN ('DONE','REVIEW_REQUIRED') "
            "ORDER BY CASE WHEN extraction IS NOT NULL THEN 0 ELSE 1 END, created_at",
            (folder_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def all_documents(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM documents ORDER BY created_at")]

    def update_document(self, identity: str, status: str, error: str = "", **values) -> None:
        allowed = {"attempts", "next_attempt", "content_sha256", "extraction"}
        if set(values) - allowed:
            raise ValueError("Unsupported document fields")
        updates = {"status": status, "error": error, "updated_at": time.time(), **values}
        sql = ",".join(f"{key}=?" for key in updates)
        with self.transaction():
            self.db.execute(f"UPDATE documents SET {sql} WHERE id=?", (*updates.values(), identity))

    def stage(self, identity: str, extraction: dict, payloads: list[dict], content_sha256: str = "") -> None:
        document = self.document(identity)
        if not document or not payloads:
            raise ValueError("A known document and nonempty payload set are required")

        keys = []
        seen_explicit_codes = set()
        for idx, payload in enumerate(payloads):
            explicit_code = str(payload.get("codigo", "")).strip()
            if explicit_code:
                if explicit_code in seen_explicit_codes:
                    raise PublicationConflict(f"Duplicate sample code '{explicit_code}' inside document")
                seen_explicit_codes.add(explicit_code)
                keys.append(explicit_code)
            else:
                ident = str(payload.get("identificacao", "")).strip()
                if not ident:
                    raise PublicationConflict("Missing both sample code and identificacao")
                digest = hashlib.sha256(f"{identity}:{idx}:{ident}".encode("utf-8")).hexdigest()[:12]
                keys.append(f"anon_{digest}")

        now = time.time()
        with self.transaction():
            for key, payload in zip(keys, payloads):
                existing = self.db.execute("SELECT * FROM samples WHERE code=?", (key,)).fetchone()
                fingerprint = payload_hash(payload)
                if existing and existing["fingerprint"] != fingerprint:
                    raise PublicationConflict(f"Sample '{key}' already staged with conflicting attributes")
                if not existing:
                    self.db.execute(
                        "INSERT INTO samples(code,payload,fingerprint,target,status,updated_at) "
                        "VALUES(?,?,?,?,'PENDING',?)",
                        (key, canonical(payload), fingerprint, document["target"], now),
                    )
                self.db.execute(
                    "INSERT OR IGNORE INTO document_samples(document_id,code) VALUES(?,?)",
                    (identity, key),
                )
            self.db.execute(
                "UPDATE documents SET status='VALIDATED',extraction=?,content_sha256=?,error='',updated_at=? "
                "WHERE id=?",
                (canonical(extraction), content_sha256, now, identity),
            )

    def samples(self, identity: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT s.* FROM samples s "
            "JOIN document_samples d ON d.code=s.code "
            "WHERE d.document_id=? ORDER BY s.code",
            (identity,),
        ).fetchall()
        return [dict(row) for row in rows]

    def sample(self, code: str) -> dict | None:
        row = self.db.execute("SELECT * FROM samples WHERE code=?", (code,)).fetchone()
        return dict(row) if row else None

    def all_samples(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM samples ORDER BY updated_at, code")]

    def update_sample(self, code: str, status: str, remote_id: str | None = None, error: str = "") -> None:
        if status == "CONFIRMED" and not remote_id:
            raise ValueError("Confirmation requires an export or target reference ID")
        with self.transaction():
            row = self.sample(code)
            if not row:
                raise ValueError("Unknown sample")
            self.db.execute(
                "UPDATE samples SET status=?,remote_id=COALESCE(?,remote_id),error=?,updated_at=? WHERE code=?",
                (status, str(remote_id) if remote_id else None, error, time.time(), code),
            )

    def event(self, identity: str | None, stage: str, **data) -> None:
        with self.transaction():
            self.db.execute(
                "INSERT INTO events(document_id,stage,created_at,data) VALUES(?,?,?,?)",
                (identity, stage, time.time(), canonical(data)),
            )

    def save_run(self, task_id: str, payload: dict, status: str = "PENDING") -> None:
        with self.transaction():
            self.db.execute(
                "INSERT INTO runs(task_id,payload,status,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload,status=excluded.status,"
                "updated_at=excluded.updated_at",
                (str(task_id), canonical(payload), status, time.time()),
            )

    def pending_runs(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM runs WHERE status='PENDING' ORDER BY updated_at")]

    def backup(self, destination: str | Path) -> None:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest_path)
        try:
            self.db.backup(target)
        finally:
            target.close()

    def queue_telemetry(self, row: dict) -> None:
        with self.transaction():
            self.db.execute("INSERT INTO telemetry(row_data) VALUES(?)", (canonical(row),))
