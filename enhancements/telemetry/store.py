"""In-memory telemetry cache for enhancement endpoints."""

from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


class TelemetryStore:
    def __init__(self, timeline_limit: int = 100) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._global: Dict[str, Any] = {}
        self._timeline_limit = timeline_limit
        self._lock = asyncio.Lock()

    async def update_request(
        self,
        path: str,
        session_id: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> None:
        if path != "/step" or payload is None:
            return
        sid = session_id or "default"
        action_payload = {k: v for k, v in payload.items() if k != "session_id"}
        async with self._lock:
            session = self._sessions.setdefault(sid, _new_session())
            session["last_action"] = copy.deepcopy(action_payload)
            session["last_action_ts"] = time.time()

    async def update_response(
        self,
        path: str,
        session_id: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> None:
        if payload is None:
            return
        if path == "/metrics":
            async with self._lock:
                self._global["metrics"] = copy.deepcopy(payload)
                self._global["metrics_ts"] = time.time()
            return

        sid = session_id or "default"
        async with self._lock:
            session = self._sessions.setdefault(sid, _new_session())
            now = time.time()
            if path == "/reset":
                session["last_reset"] = copy.deepcopy(payload)
                session["last_state"] = copy.deepcopy(payload)
                session["last_state_ts"] = now
            elif path == "/state":
                session["last_state"] = copy.deepcopy(payload)
                session["last_state_ts"] = now
            elif path == "/trajectory":
                session["last_trajectory"] = copy.deepcopy(payload)
                session["last_trajectory_ts"] = now
            elif path == "/step":
                session["last_step"] = copy.deepcopy(payload)
                session["last_step_ts"] = now
                _append_timeline(session, payload, now, self._timeline_limit)

    async def snapshot(self, session_id: Optional[str]) -> Dict[str, Any]:
        sid = session_id or "default"
        async with self._lock:
            session = copy.deepcopy(self._sessions.get(sid, _new_session()))
            global_metrics = copy.deepcopy(self._global)
        return {
            "session_id": sid,
            "state": session.get("last_state"),
            "last_step": session.get("last_step"),
            "trajectory": session.get("last_trajectory"),
            "timeline": list(session.get("timeline", [])),
            "last_action": session.get("last_action"),
            "global_metrics": global_metrics.get("metrics"),
            "timestamps": {
                "state": session.get("last_state_ts"),
                "step": session.get("last_step_ts"),
                "trajectory": session.get("last_trajectory_ts"),
                "metrics": global_metrics.get("metrics_ts"),
            },
        }


telemetry_store = TelemetryStore()


def _new_session() -> Dict[str, Any]:
    return {
        "timeline": deque(),
    }


def _append_timeline(
    session: Dict[str, Any],
    payload: Dict[str, Any],
    ts: float,
    limit: int,
) -> None:
    timeline: Deque[Dict[str, Any]] = session.setdefault("timeline", deque())
    entry = {
        "ts": ts,
        "reward": payload.get("reward"),
        "done": payload.get("done"),
        "info": payload.get("info"),
    }
    timeline.append(entry)
    while len(timeline) > limit:
        timeline.popleft()
