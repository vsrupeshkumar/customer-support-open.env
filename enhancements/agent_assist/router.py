"""Agent assist endpoints."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from enhancements.agent_assist.service import explain
from enhancements.telemetry.store import telemetry_store
from enhancements.utils.perf_guard import get_perf_guard

router = APIRouter()


class ExplainRequest(BaseModel):
    session_id: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


class ExplainResponse(BaseModel):
    request_id: str
    status: str


_explain_store: Dict[str, Dict[str, Any]] = {}
_store_lock = asyncio.Lock()


@router.post("/explain", response_model=ExplainResponse, tags=["enhanced"])
async def explain_action(payload: ExplainRequest) -> ExplainResponse:
    guard = get_perf_guard()
    if guard and guard.disabled:
        raise HTTPException(
            status_code=503,
            detail=f"Enhancements disabled: {guard.disabled_reason}",
        )

    snapshot = await telemetry_store.snapshot(payload.session_id)
    state = payload.state or snapshot.get("state")
    action = payload.action or snapshot.get("last_action")
    reward = payload.reward or (snapshot.get("last_step") or {}).get("reward")
    info = payload.info or (snapshot.get("last_step") or {}).get("info")

    request_id = str(uuid.uuid4())
    async with _store_lock:
        _explain_store[request_id] = {
            "status": "queued",
            "created_ts": time.time(),
        }

    asyncio.create_task(
        _run_explain(request_id, {
            "state": state,
            "action": action,
            "reward": reward,
            "info": info,
        })
    )

    return ExplainResponse(request_id=request_id, status="queued")


@router.get("/explain/{request_id}", tags=["enhanced"])
async def explain_result(request_id: str) -> Dict[str, Any]:
    async with _store_lock:
        result = _explain_store.get(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Explain request not found")
    return result


async def _run_explain(request_id: str, payload: Dict[str, Any]) -> None:
    try:
        result = await explain(payload)
        status = "ready"
    except Exception as exc:
        result = {"error": str(exc)}
        status = "error"

    async with _store_lock:
        _explain_store[request_id] = {
            "status": status,
            "result": result,
            "completed_ts": time.time(),
        }
