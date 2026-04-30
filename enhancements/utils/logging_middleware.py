"""Middleware for safe, non-blocking enhancement logging and telemetry."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from enhancements.telemetry.store import TelemetryStore
from enhancements.utils.perf_guard import PerfGuard


_TARGET_PATHS = {"/reset", "/step", "/state", "/trajectory", "/metrics"}


class EnhancementMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        store: Optional[TelemetryStore],
        perf_guard: Optional[PerfGuard],
        log_path: str,
        enable_logging: bool = True,
        enable_telemetry: bool = True,
    ) -> None:
        super().__init__(app)
        self._store = store
        self._perf_guard = perf_guard
        self._log_path = log_path
        self._enable_logging = enable_logging
        self._enable_telemetry = enable_telemetry

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/enhanced"):
            return await call_next(request)

        if self._perf_guard and self._perf_guard.disabled:
            return await call_next(request)

        start = time.perf_counter()
        request_payload = None
        session_id = None

        if self._enable_telemetry and path in _TARGET_PATHS:
            if path in {"/step", "/reset"}:
                request_payload = await self._read_body(request)
                session_id = _get_session_id_from_payload(request_payload)
                if self._store:
                    await self._store.update_request(path, session_id, request_payload)
            elif path in {"/state", "/trajectory"}:
                session_id = request.query_params.get("session_id")

        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        if self._perf_guard:
            await self._perf_guard.record(latency_ms)

        if self._enable_logging:
            asyncio.create_task(
                _write_log_entry(
                    self._log_path,
                    {
                        "ts": time.time(),
                        "path": path,
                        "method": request.method,
                        "status": response.status_code,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
            )

        if not (self._enable_telemetry and self._store and path in _TARGET_PATHS):
            return response

        response_body = await _consume_body(response)
        response_payload = _safe_json(response_body)
        if path == "/metrics" and self._store:
            await self._store.update_response(path, None, response_payload)
        else:
            await self._store.update_response(path, session_id, response_payload)

        headers = dict(response.headers)
        headers.pop("content-length", None)
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )

    async def _read_body(self, request: Request) -> Optional[Dict[str, Any]]:
        try:
            body = await request.body()
            request._body = body  # noqa: SLF001 - preserve body for downstream
            return _safe_json(body)
        except Exception:
            return None


async def _consume_body(response: Response) -> bytes:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    return body


def _safe_json(body: bytes) -> Optional[Dict[str, Any]]:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _get_session_id_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    session_id = payload.get("session_id")
    if session_id:
        return str(session_id)
    return None


async def _write_log_entry(path: str, entry: Dict[str, Any]) -> None:
    try:
        line = json.dumps(entry, ensure_ascii=True)
        await asyncio.to_thread(_append_line, path, line)
    except Exception:
        return


def _append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
