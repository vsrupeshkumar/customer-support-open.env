"""Feature flags and configuration for optional enhancements."""

from __future__ import annotations

import os
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_ENHANCEMENTS: bool = _env_bool("ENABLE_ENHANCEMENTS", True)
ENABLE_TELEMETRY: bool = _env_bool("ENABLE_TELEMETRY", True)
ENABLE_AGENT_ASSIST: bool = _env_bool("ENABLE_AGENT_ASSIST", False)
ENABLE_VISUALIZATION: bool = _env_bool("ENABLE_VISUALIZATION", True)
ENABLE_ENHANCED_LOGGING: bool = _env_bool("ENABLE_ENHANCED_LOGGING", True)

ENHANCEMENT_LOG_PATH: str = os.getenv("ENHANCED_LOG_PATH", "/tmp/enhanced_logs.json")

_base_ms = os.getenv("ENHANCEMENT_BASELINE_MS")
ENHANCEMENT_BASELINE_MS: Optional[float] = float(_base_ms) if _base_ms else None
ENHANCEMENT_MAX_LATENCY_MS: float = float(os.getenv("ENHANCEMENT_MAX_LATENCY_MS", "1200"))
ENHANCEMENT_LATENCY_WINDOW: int = int(os.getenv("ENHANCEMENT_LATENCY_WINDOW", "20"))
ENHANCEMENT_LATENCY_RATIO: float = float(os.getenv("ENHANCEMENT_LATENCY_RATIO", "1.05"))
