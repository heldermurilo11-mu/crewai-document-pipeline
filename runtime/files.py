"""Atomic JSON/text persistence and single-instance advisory file locking."""

import json
import os
from pathlib import Path
import tempfile

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class AlreadyRunning(RuntimeError):
    """Raised when another active instance already holds the runtime lock."""


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Write text via sibling temp file + fsync + atomic os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        return target
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_json(path: str | Path, data) -> Path:
    """Serialize *data* to JSON atomically."""
    return atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


class InstanceLock:
    """Cross-platform non-blocking file lock preventing overlapping runs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "InstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                self._handle.seek(0)
                if self.path.stat().st_size == 0:
                    self._handle.write(b"\x00")
                    self._handle.flush()
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.write(f"pid={os.getpid()}\n".encode("ascii"))
            self._handle.flush()
            return self
        except OSError as exc:
            self._close_handle()
            raise AlreadyRunning(f"Pipeline lock held at {self.path}") from exc

    def _close_handle(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._close_handle()
