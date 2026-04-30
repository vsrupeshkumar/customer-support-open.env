"""Latency guardrail for optional enhancement layer."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional


class PerfGuard:
    def __init__(
        self,
        baseline_ms: Optional[float] = None,
        max_latency_ms: float = 1200.0,
        window: int = 20,
        ratio: float = 1.05,
    ) -> None:
        self._baseline_ms = baseline_ms
        self._max_latency_ms = max_latency_ms
        self._ratio = ratio
        self._window = max(1, window)
        self._samples = deque(maxlen=self._window)
        self._disabled = False
        self._disabled_reason: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    async def record(self, latency_ms: float) -> None:
        async with self._lock:
            if self._disabled:
                return
            self._samples.append(float(latency_ms))
            if len(self._samples) < self._window:
                return

            avg = sum(self._samples) / len(self._samples)
            if self._baseline_ms is None:
                self._baseline_ms = avg
                if avg > self._max_latency_ms:
                    self._disable(avg, self._max_latency_ms)
                return

            threshold = min(self._max_latency_ms, self._baseline_ms * self._ratio)
            if avg > threshold:
                self._disable(avg, threshold)

    def _disable(self, avg: float, threshold: float) -> None:
        self._disabled = True
        self._disabled_reason = (
            f"latency_avg_ms={avg:.2f} threshold_ms={threshold:.2f}"
        )


_perf_guard: Optional[PerfGuard] = None


def set_perf_guard(guard: PerfGuard) -> None:
    global _perf_guard
    _perf_guard = guard


def get_perf_guard() -> Optional[PerfGuard]:
    return _perf_guard
