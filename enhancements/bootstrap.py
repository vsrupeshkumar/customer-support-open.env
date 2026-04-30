"""Enhancement layer bootstrapper."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from enhancements.agent_assist.router import router as agent_assist_router
from enhancements.config.flags import (
    ENABLE_AGENT_ASSIST,
    ENABLE_ENHANCEMENTS,
    ENABLE_ENHANCED_LOGGING,
    ENABLE_TELEMETRY,
    ENABLE_VISUALIZATION,
    ENHANCEMENT_BASELINE_MS,
    ENHANCEMENT_LATENCY_RATIO,
    ENHANCEMENT_LATENCY_WINDOW,
    ENHANCEMENT_LOG_PATH,
    ENHANCEMENT_MAX_LATENCY_MS,
)
from enhancements.telemetry.router import router as telemetry_router
from enhancements.telemetry.store import telemetry_store
from enhancements.utils.logging_middleware import EnhancementMiddleware
from enhancements.utils.perf_guard import PerfGuard, set_perf_guard

logger = logging.getLogger("crisis_env.enhancements")


def init_enhancements(app: FastAPI) -> None:
    if not ENABLE_ENHANCEMENTS:
        logger.info("Enhancements disabled by flag.")
        return

    guard = PerfGuard(
        baseline_ms=ENHANCEMENT_BASELINE_MS,
        max_latency_ms=ENHANCEMENT_MAX_LATENCY_MS,
        window=ENHANCEMENT_LATENCY_WINDOW,
        ratio=ENHANCEMENT_LATENCY_RATIO,
    )
    set_perf_guard(guard)

    enable_telemetry = ENABLE_TELEMETRY or ENABLE_AGENT_ASSIST
    app.add_middleware(
        EnhancementMiddleware,
        store=telemetry_store if enable_telemetry else None,
        perf_guard=guard,
        log_path=ENHANCEMENT_LOG_PATH,
        enable_logging=ENABLE_ENHANCED_LOGGING,
        enable_telemetry=enable_telemetry,
    )

    if ENABLE_TELEMETRY:
        app.include_router(telemetry_router, prefix="/enhanced")

    if ENABLE_AGENT_ASSIST:
        app.include_router(agent_assist_router, prefix="/enhanced")

    if ENABLE_VISUALIZATION:
        web_dir = Path(__file__).parent / "visualization" / "web"
        if web_dir.exists():
            app.mount(
                "/enhanced/ui",
                StaticFiles(directory=str(web_dir), html=True),
                name="enhanced-ui",
            )
            logger.info("Enhanced UI mounted at /enhanced/ui (dir=%s)", web_dir)
        else:
            logger.info("Enhanced UI directory not found at %s", web_dir)
