"""
server.py
=========
Minimal FastAPI wrapper that exposes the CrisisManagementEnv as a
Hugging Face-compatible HTTP API service.

Session Architecture
--------------------
UUID-based session isolation prevents state bleeding between concurrent
agents during Phase 2 auto-evaluation.  Each ``POST /reset`` creates a
new isolated ``CrisisManagementEnv`` instance keyed by a UUID.  Subsequent
``POST /step`` and ``GET /state`` requests reference the session via
``session_id`` in the request body/query.

Backward Compatibility
----------------------
If no ``session_id`` is provided, the server uses a default ``"default"``
session — ensuring existing ``inference.py`` scripts work with zero
code changes.  Total active sessions are capped at ``MAX_SESSIONS``
to ensure memory safety.
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
import math
import random
import secrets
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

# Load environment variables from .env file
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from env import CrisisManagementEnv
from env.models import Action, EnvironmentState, Observation, StructuralHallucinationError, PoisonAction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] SERVER - %(message)s",
)
logger = logging.getLogger("crisis_env.server")

def log_event(tag: str, message: dict):
    """Strict M2M key=value emitter for START/END grader markers (server-side)."""
    kv = " ".join(f"{k}={v}" for k, v in message.items())
    print(f"[{tag}] {kv}", flush=True)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive Crisis Management Environment",
    description=(
        "OpenEnv-compliant multi-zone emergency response RL environment. "
        "Exposes reset / step / state over HTTP for Hugging Face evaluation. "
        "Supports UUID-based session isolation for concurrent agent evaluation."
    ),
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session Store (replaces global _env singleton)
# ---------------------------------------------------------------------------
# Mathematical Detail: N_sessions ≤ MAX_CAPACITY to ensure memory safety.

MAX_SESSIONS: int = 16

_sessions: Dict[str, CrisisManagementEnv] = {}
_session_lock: asyncio.Lock = asyncio.Lock()

# Backward compatibility: module-level reference for the validation handler
_env: Optional[CrisisManagementEnv] = None


def _get_session(session_id: Optional[str]) -> CrisisManagementEnv:
    """Look up a session by ID, falling back to 'default' if not provided.

    Args:
        session_id: UUID session identifier, or None for the default session.

    Returns:
        The CrisisManagementEnv instance for the given session.

    Raises:
        HTTPException: 400 if no session exists for the given ID.
    """
    sid = session_id or "default"
    env = _sessions.get(sid)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{sid}' not found. Call POST /reset first.",
        )
    return env


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Attempt to extract session ID from the invalid payload
    try:
        body_bytes = await request.body()
        data = json.loads(body_bytes) if body_bytes else {}
        session_id = data.get("session_id")
    except Exception:
        session_id = None

    try:
        env = _get_session(session_id)
    except HTTPException:
        return JSONResponse(
            status_code=400,
            content={"detail": "Environment not initialised. Call POST /reset first."}
        )
    
    # Process the structural hallucination through the MDP engine properly.
    # This invokes all POMDP scaling, resource cooldown logic, and penalty processing,
    # rather than duplicating arithmetic incorrectly in the API layer.
    action = StructuralHallucinationError("LLM generated an invalid JSON schema.")
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        success = info.get("resolved", 0) == info.get("total", 0)
        log_event("END", {"success": str(success).lower(), "score": info.get("score", 0.0)})
        
        # Medium 4.1: Update Global Metrics
        _global_metrics["episodes_completed"] += 1.0
        _global_metrics["total_reward_all"] += float(info.get("score", 0.0))
        if success:
            _global_metrics["episodes_succeeded"] += 1.0
            
        # Auto-cleanup: remove completed sessions (except default)
        if session_id and session_id != "default":
            async with _session_lock:
                _sessions.pop(session_id, None)
                logger.info("Session %s auto-cleaned after hallucination.", session_id)
                
    # Force the 200 OK with the exact StepResponse schema
    step_resp = StepResponse(
        observation=obs.model_dump(mode="json"),
        reward=float(reward),
        done=done,
        info=info
    )
    
    return JSONResponse(status_code=200, content=step_resp.model_dump())

class StepResponse(BaseModel):
    """Response payload for POST /step."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status: str
    active_sessions: int
    max_sessions: int
    groq_api_reachable: bool
    memory_rss_mb: float

# Medium 4.1: Production Observability Endpoints
_global_metrics: Dict[str, float] = {
    "episodes_completed": 0.0,
    "episodes_succeeded": 0.0,
    "total_reward_all": 0.0,
}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe with session capacity metrics and API reachability.

    Returns:
        Status, active session count, maximum session capacity, memory, and API reachability.
    """
    import os
    import requests
    
    # Check Groq API Reachability
    try:
        res = requests.get("https://api.groq.com/openai/v1/models", timeout=2.0)
        groq_reachable = res.status_code in (200, 401, 403)
    except Exception:
        groq_reachable = False

    # Check Memory Usage using OS proc files
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
            mem_mb = (rss_pages * os.sysconf('SC_PAGE_SIZE')) / (1024 * 1024)
    except Exception:
        mem_mb = 0.0

    return HealthResponse(
        status="ok",
        active_sessions=len(_sessions),
        max_sessions=MAX_SESSIONS,
        groq_api_reachable=groq_reachable,
        memory_rss_mb=round(mem_mb, 2),
    )

@app.get("/metrics", tags=["meta"])
async def metrics() -> Dict[str, Any]:
    """Production episode statistics."""
    c = _global_metrics["episodes_completed"]
    mean_reward = _global_metrics["total_reward_all"] / c if c > 0 else 0.0
    completion_rate = _global_metrics["episodes_succeeded"] / c if c > 0 else 0.0
    return {
        "episodes_completed": int(c),
        "mean_reward": round(mean_reward, 4),
        "completion_rate": round(completion_rate, 4),
    }

@app.post("/reset", response_model=Dict[str, Any], tags=["openenv"])
async def reset(request: Request) -> Dict[str, Any]:
    global _env
    try:
        data = await request.json()
        task_id = int(data.get("task_id", 1))
        seed = data.get("seed")
        session_id = data.get("session_id")  # Optional: client-provided session ID
        if seed is not None:
            seed = int(seed)
        else:
            seed = secrets.randbelow(100000) + 1
    except Exception:
        # Fallback if the body is missing or malformed to avoid 422 errors
        task_id = 1
        seed = secrets.randbelow(100000) + 1
        session_id = None

    try:
        # Generate session ID if not provided (backward compatibility)
        if not session_id:
            session_id = "default"

        # Capacity guard: reject new sessions if at capacity
        async with _session_lock:
            if session_id not in _sessions and len(_sessions) >= MAX_SESSIONS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Session capacity reached ({MAX_SESSIONS}). "
                           f"Active sessions: {list(_sessions.keys())}",
                )
            new_env = CrisisManagementEnv(task_id=task_id, seed=seed)
            _sessions[session_id] = new_env
            # Backward compatibility: update module-level _env for validation handler
            _env = new_env

        obs = new_env.obs  # Access the cleanly initialized state
        logger.info(
            "Environment initialized: task_id=%d seed=%s session_id=%s (active=%d/%d)",
            task_id, seed, session_id, len(_sessions), MAX_SESSIONS,
        )

        # The Mathematical Standout: State Entropy Calculation
        zones = obs.zones.values()
        n = max(len(zones), 1)
        state_counts = {}
        for z in zones:
            s_combo = (z.fire.value, z.patient.value, z.traffic.value)
            state_counts[s_combo] = state_counts.get(s_combo, 0) + 1
        
        entropy = 0.0
        for count in state_counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
                
        obs_dict = obs.model_dump(mode="json")
        obs_dict["session_id"] = session_id

        # Log EVENT START
        log_event("START", {"task": task_id, "seed": seed, "session": session_id, "entropy": round(entropy, 4)})
        
        return obs_dict
    except HTTPException:
        raise  # Re-raise capacity guard errors
    except Exception as exc:
        logger.exception("Error during /reset: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/step", response_model=StepResponse, tags=["openenv"])
async def step(request: Request) -> StepResponse:
    """Pure dumb router — POMDP math lives exclusively in env/reward.py.

    Resilience contract: this endpoint NEVER returns HTTP 5xx to the agent.
    Any internal error (Reward ledger validation, Pydantic mismatch, etc.)
    is caught and converted to a safe -5.0 penalty StepResponse so the
    inference script's raise_for_status() never aborts an episode mid-run.

    Session Isolation: Extracts ``session_id`` from the request body.
    Falls back to ``"default"`` if not provided (backward compatibility).
    """
    try:
        data = await request.json()
        session_id = data.pop("session_id", None)  # Extract and remove before Action parse
        action = Action(**data)
    except Exception as e:
        session_id = None
        action = PoisonAction(error_msg=str(e))

    env = _get_session(session_id)

    try:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        logger.info("step session=%s reward=%.4f done=%s", session_id or "default", reward, done)

        async with _session_lock:
            if done:
                success = info.get("resolved", 0) == info.get("total", 0)
                log_event("END", {"success": str(success).lower(), "score": info.get("score", 0.0)})
                
                # Medium 4.1: Update Global Metrics
                _global_metrics["episodes_completed"] += 1.0
                _global_metrics["total_reward_all"] += float(info.get("score", 0.0))
                if success:
                    _global_metrics["episodes_succeeded"] += 1.0
                    
                # Auto-cleanup: remove completed sessions (except default)
                if session_id and session_id != "default":
                    _sessions.pop(session_id, None)
                    logger.info("Session %s auto-cleaned after episode end.", session_id)

        return StepResponse(
            observation=obs.model_dump(mode="json"),
            reward=float(reward),
            done=done,
            info=info,
        )
    except Exception as exc:
        # Internal environment error (reward ledger, Pydantic, etc.).
        # Return a safe penalty response so the agent episode continues.
        logger.exception("[ENV INTERNAL ERROR] /step failed: %s", exc)
        safe_obs = env.obs.model_dump(mode="json") if env is not None else {}
        return StepResponse(
            observation=safe_obs,
            reward=-5.0,
            done=False,
            info={"error": f"Internal environment error: {exc}"},
        )

@app.get("/state", response_model=Dict[str, Any], tags=["openenv"])
async def state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the full internal environment state for a session.

    Args:
        session_id: Optional query parameter for session identification.
    """
    env = _get_session(session_id)
    try:
        env_state: EnvironmentState = env.state
        return env_state.model_dump(mode="json")
    except Exception as exc:
        logger.exception("Error during /state: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.delete("/session/{session_id}", tags=["meta"])
async def delete_session(session_id: str) -> Dict[str, str]:
    """Manually delete a session to free resources.

    Args:
        session_id: The UUID of the session to delete.
    """
    async with _session_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info("Session %s manually deleted.", session_id)
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
