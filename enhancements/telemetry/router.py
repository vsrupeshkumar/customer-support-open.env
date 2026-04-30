"""Telemetry endpoints for enhanced read-only insights."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from enhancements.telemetry.store import telemetry_store
from enhancements.utils.perf_guard import get_perf_guard

router = APIRouter()


@router.get("/metrics", tags=["enhanced"])
async def enhanced_metrics(session_id: Optional[str] = None) -> Dict[str, Any]:
    guard = get_perf_guard()
    if guard and guard.disabled:
        raise HTTPException(
            status_code=503,
            detail=f"Enhancements disabled: {guard.disabled_reason}",
        )

    snapshot = await telemetry_store.snapshot(session_id)
    return {
        "telemetry": snapshot,
    }
