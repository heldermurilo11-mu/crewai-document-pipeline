"""Offline tests for OS-level subprocess containment, log bounding, and supervisor watchdogs."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from runtime.config import RuntimeConfig
from runtime.deadline import DeadlineExceeded
from runtime.supervisor import (
    WorkerFailure,
    _stop_registered_groups,
    run_worker,
    supervise_command,
)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_worker_returns_complete_atomic_result(self):
        value = run_worker("tests.workers:result", {"value": 42}, 10, self.path)
        self.assertEqual(value, {"value": 42})
        self.assertEqual(list(self.path.glob("*.json")), [])

    def test_linux_registry_protects_reused_pid(self):
        for pid in (10, 20, 30):
            (self.path / f"worker-{pid}.json").write_text(
                json.dumps({"pid": pid, "start": "original"}), encoding="utf-8"
            )
        identities = {10: "original", 20: "reused", 30: None}
        with patch("runtime.supervisor._proc_start", side_effect=identities.get), patch(
            "runtime.supervisor._kill_group"
        ) as stop:
            _stop_registered_groups(self.path)
        self.assertEqual(sorted(call.args[0] for call in stop.call_args_list), [10, 30])
        self.assertEqual(list(self.path.glob("worker-*.json")), [])

    def test_blocked_worker_is_killed_and_next_worker_runs(self):
        started = time.monotonic()
        with self.assertRaises(DeadlineExceeded):
            run_worker("tests.workers:blocked", {}, 0.7, self.path, grace=1)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(run_worker("tests.workers:result", {"value": "next"}, 10, self.path)["value"], "next")

    def test_worker_failure_is_explicit(self):
        with self.assertRaises(WorkerFailure):
            run_worker("tests.workers:failure", {}, 10, self.path)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object validation")
    def test_worker_kills_descendants_and_preserves_unrelated_process(self):
        import ctypes
        from ctypes import wintypes

        marker = self.path / "descendant.pid"
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with self.assertRaises(DeadlineExceeded):
                run_worker("tests.workers:descendants", {"marker": str(marker)}, 2, self.path, grace=1)
            self.assertIsNone(unrelated.poll())
            self.assertTrue(marker.exists(), "Worker must spawn descendant before deadline")
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, int(marker.read_text(encoding="utf-8")))
            if handle:
                try:
                    code = wintypes.DWORD()
                    self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                    self.assertNotEqual(code.value, 259, "Owned descendant stayed alive")
                finally:
                    kernel.CloseHandle(handle)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_large_stdout_does_not_deadlock_or_fill_disk(self):
        self.assertTrue(run_worker("tests.workers:chatter", {}, 10, self.path)["complete"])
        self.assertLessEqual(max(p.stat().st_size for p in self.path.glob("*.log")), 2 * 1024 * 1024)

    def test_outer_supervisor_recovers_hung_controller(self):
        config = replace(
            RuntimeConfig(),
            state_dir=str(self.path),
            batch_budget_seconds=0.5,
            supervisor_grace_seconds=0.2,
            termination_grace_seconds=1.0,
        )
        marker = self.path / "context.json"
        code = supervise_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            config,
            context_path=marker,
        )
        self.assertEqual(code, 124)
        self.assertTrue(json.loads(marker.read_text(encoding="utf-8"))["process_terminated"])
        next_code = supervise_command(
            [sys.executable, "-c", "print('next run clean')"],
            replace(config, batch_budget_seconds=5.0),
        )
        self.assertEqual(next_code, 0)


if __name__ == "__main__":
    unittest.main()
