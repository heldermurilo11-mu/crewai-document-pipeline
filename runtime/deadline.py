"""Monotonic wall-clock deadline propagation and timeout clamping."""

import time


class DeadlineExceeded(TimeoutError):
    """Raised when an operation exhausts its remaining wall-clock budget."""


class Deadline:
    """Tracks remaining wall-clock seconds using ``time.monotonic``."""

    def __init__(self, total_seconds: float, started_monotonic: float | None = None):
        if total_seconds <= 0:
            raise ValueError("Budget must be positive")
        self.total_seconds = float(total_seconds)
        self.started_monotonic = (
            time.monotonic() if started_monotonic is None else float(started_monotonic)
        )

    @property
    def remaining(self) -> float:
        return max(0.0, (self.started_monotonic + self.total_seconds) - time.monotonic())

    def check(self, operation: str = "operation", reserve: float = 0.0) -> float:
        left = self.remaining
        if left <= reserve:
            raise DeadlineExceeded(
                f"Deadline exhausted before {operation}: remaining={left:.2f}s reserve={reserve:.2f}s"
            )
        return left

    def timeout(self, configured_seconds: float, reserve: float = 0.0, minimum: float = 0.1) -> float:
        left = self.check("network request", reserve=reserve)
        return max(float(minimum), min(float(configured_seconds), left - reserve))

    def timeout_ms(self, configured_ms: int, reserve: float = 0.0, minimum_ms: int = 100) -> int:
        left = self.check("HTTP request", reserve=reserve)
        available_ms = int((left - reserve) * 1000)
        return max(int(minimum_ms), min(int(configured_ms), available_ms))

    def sleep(self, seconds: float, reserve: float = 0.0) -> None:
        left = self.check("retry wait", reserve=reserve)
        pause = min(float(seconds), max(0.0, left - reserve))
        if pause > 0:
            time.sleep(pause)
