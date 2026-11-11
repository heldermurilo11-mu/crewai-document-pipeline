"""OS-level subprocess containment, bounded logging, and wall-clock watchdog."""

import ctypes
from ctypes import wintypes
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .deadline import DeadlineExceeded
from .files import atomic_json


class WorkerFailure(RuntimeError):
    """Raised when an isolated file worker exits with an unhandled error."""


_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _proc_start(pid: int) -> str | None:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return None
    try:
        raw = stat_path.read_text(encoding="utf-8", errors="ignore")
        tail = raw[raw.rfind(")") + 2 :].split()
        return tail[19] if len(tail) > 19 else None
    except OSError:
        return None


def _kill_group(pid: int) -> None:
    if os.name == "nt":
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _stop_registered_groups(registry: Path) -> None:
    for marker in list(Path(registry).glob("worker-*.json")):
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(info["pid"])
            expected_start = info.get("start")
            current_start = _proc_start(pid)
            if current_start is None or expected_start is None or current_start == expected_start:
                _kill_group(pid)
        except Exception:
            pass
        finally:
            marker.unlink(missing_ok=True)


class WindowsJob:
    """Assigns child processes to a Win32 Job Object terminated on handle close."""

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel = kernel
        kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_uint64)] * 6

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            self.handle,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, pid: int) -> None:
        if os.name != "nt" or not self.handle:
            return
        # PROCESS_SET_QUOTA (0x0100) | PROCESS_TERMINATE (0x0001) | SYNCHRONIZE (0x00100000)
        process = self.kernel.OpenProcess(0x0101 | 0x00100000, False, int(pid))
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def terminate(self) -> None:
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 124):
            err = ctypes.get_last_error()
            if err:
                raise ctypes.WinError(err)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class LimitedLog:
    """Bounds child stdout/stderr log files to prevent disk exhaustion."""

    def __init__(self, stream, limit: int = 2 * 1024 * 1024):
        self.stream = stream
        self.remaining = limit

    def write(self, text: str) -> int:
        raw = str(text)
        if self.remaining > 0:
            encoded = raw.encode("utf-8")[: self.remaining]
            part = encoded.decode("utf-8", errors="ignore")
            self.stream.write(part)
            self.remaining -= len(part.encode("utf-8"))
        return len(raw)

    def flush(self) -> None:
        self.stream.flush()


def _worker_entry(target: str, arguments: dict, result_path: str, log_path: str, gate) -> None:
    if not gate.wait(10):
        return
    if os.name != "nt":
        os.setsid()
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = LimitedLog(log)
        try:
            module, function = target.split(":", 1)
            fn = getattr(importlib.import_module(module), function)
            result = fn(**arguments)
            atomic_json(result_path, {"ok": True, "result": result})
        except BaseException as exc:
            atomic_json(
                result_path,
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
            )
        finally:
            log.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr


def _stop_mp(process, job: WindowsJob, grace: float) -> None:
    if os.name == "nt":
        job.terminate()
    else:
        _kill_group(process.pid)
        if process.is_alive():
            process.kill()
    process.join(grace)
    if process.is_alive():
        process.kill()
        process.join(grace)
    if process.is_alive():
        raise RuntimeError("Worker termination could not be confirmed")


def run_worker(target: str, arguments: dict, seconds: float, directory: str | Path, grace: float = 5):
    """Run *target* (`module:fn`) in an isolated spawned process under strict timeout."""
    if seconds <= 0:
        raise DeadlineExceeded("No remaining worker budget")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    result_path = directory / f"{token}.json"
    log_path = directory / f"{token}.log"

    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    process = context.Process(
        target=_worker_entry,
        args=(target, arguments, str(result_path), str(log_path), gate),
    )
    registry_marker = None
    job = WindowsJob()
    started = time.monotonic()
    try:
        process.start()
        job.assign(process.pid)
        if os.name != "nt" and os.getenv("PIPELINE_OWNED_WORKERS"):
            registry_marker = Path(os.environ["PIPELINE_OWNED_WORKERS"]) / f"worker-{token}.json"
            atomic_json(registry_marker, {"pid": process.pid, "start": _proc_start(process.pid)})
        gate.set()
        process.join(max(0.0, seconds - (time.monotonic() - started)))
        if process.is_alive():
            _stop_mp(process, job, grace)
            raise DeadlineExceeded("File worker exceeded its wall-clock deadline")
        if process.exitcode != 0 or not result_path.is_file():
            raise WorkerFailure(f"Worker exited without a complete result (exit={process.exitcode})")
        if result_path.stat().st_size > 64 * 1024 * 1024:
            raise WorkerFailure("Worker result exceeds 64 MB safety limit")
        envelope = json.loads(result_path.read_text(encoding="utf-8"))
        if not envelope.get("ok"):
            if envelope.get("error_type") == "DeadlineExceeded":
                raise DeadlineExceeded(envelope.get("error") or "Worker exhausted its deadline")
            err_type = envelope.get("error_type", "UnknownError")
            err_msg = envelope.get("error", "")
            raise WorkerFailure(f"{err_type}: {err_msg}" if err_msg else err_type)
        return envelope["result"]
    finally:
        if process.pid and process.is_alive():
            _stop_mp(process, job, grace)
        job.close()
        if process.pid:
            if os.name != "nt":
                _kill_group(process.pid)
            process.close()
        if registry_marker and registry_marker.exists():
            registry_marker.unlink(missing_ok=True)
        if result_path.exists():
            result_path.unlink(missing_ok=True)


def wait_start_gate() -> None:
    gate = os.environ.get("PIPELINE_START_GATE")
    if not gate:
        return
    stop = time.monotonic() + 10.0
    while not Path(gate).exists():
        if time.monotonic() >= stop:
            raise RuntimeError("Supervisor did not release the process start gate")
        time.sleep(0.02)


def supervise_command(command, config, *, environ=None, context_path=None) -> int:
    """Supervise *command* in an OS-contained group with batch wall-clock hard kill."""
    directory = config.resolved_state_dir() / "watchdog"
    directory.mkdir(parents=True, exist_ok=True)
    gate = directory / f"start-{uuid.uuid4().hex}"
    context_path = Path(context_path or directory / f"run-{uuid.uuid4().hex}.json")
    environment = dict(os.environ if environ is None else environ)
    started = time.monotonic()
    environment.update(
        PIPELINE_SUPERVISED="1",
        PIPELINE_START_GATE=str(gate),
        PIPELINE_STARTED_MONOTONIC=str(started),
        PIPELINE_CONTEXT_PATH=str(context_path),
        PIPELINE_GUARD_STATE_DIR=str(config.resolved_state_dir()),
        PIPELINE_GUARD_BUDGET=str(config.batch_budget_seconds),
    )
    registry = None
    if os.name != "nt":
        registry = directory / f"groups-{uuid.uuid4().hex}"
        registry.mkdir()
        environment["PIPELINE_OWNED_WORKERS"] = str(registry)

    job = WindowsJob()
    process = None
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=os.name != "nt",
        )
        job.assign(process.pid)
        gate.touch()
        timeout = max(
            0.001,
            config.batch_budget_seconds + config.supervisor_grace_seconds - (time.monotonic() - started),
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                job.terminate()
            else:
                _kill_group(process.pid)
            process.wait(timeout=config.termination_grace_seconds)
            if registry:
                _stop_registered_groups(registry)
            context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {}
            context.update(
                status="SUPERVISOR_TIMEOUT",
                elapsed_seconds=time.monotonic() - started,
                process_terminated=True,
            )
            atomic_json(context_path, context)
            return 124
    finally:
        if process and process.poll() is None:
            if os.name == "nt":
                job.terminate()
            else:
                _kill_group(process.pid)
            process.wait(timeout=config.termination_grace_seconds)
        if registry:
            if process:
                _kill_group(process.pid)
            _stop_registered_groups(registry)
            registry.rmdir()
        job.close()
        if gate.exists():
            gate.unlink(missing_ok=True)
