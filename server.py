"""
server.py
=========
Minimal FastAPI wrapper that exposes the CrisisManagementEnv as a
Hugging Face–compatible HTTP API service.

Endpoints
---------
POST /reset   — Reset environment, returns initial Observation.
POST /step    — Execute one action, returns Observation + reward + done + info.
GET  /state   — Return the current full EnvironmentState.
GET  /health  — Liveness probe (returns {"status": "ok"}).

Run
---
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import CrisisManagementEnv
from env.models import Action, EnvironmentState, Observation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] SERVER - %(message)s",
)
logger = logging.getLogger("crisis_env.server")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive Crisis Management Environment",
    description=(
        "OpenEnv-compliant multi-zone emergency response RL environment. "
        "Exposes reset / step / state over HTTP for Hugging Face evaluation."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global environment instance (one episode at a time, server-scoped)
# ---------------------------------------------------------------------------

_env: Optional[CrisisManagementEnv] = None


def _get_env() -> CrisisManagementEnv:
    """Return the current environment, raising 400 if it hasn't been reset yet."""
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Payload accepted by POST /reset."""

    seed: Optional[int] = Field(
        default=None,
        description="Optional integer seed for deterministic episode generation.",
    )
    task_id: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Task difficulty level: 1 = easy, 2 = medium, 3 = hard.",
    )


class StepResponse(BaseModel):
    """Response payload for POST /step."""

    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe. Always returns ``{"status": "ok"}``."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=Dict[str, Any], tags=["openenv"])
async def reset(request: ResetRequest) -> Dict[str, Any]:
    """Reset the environment to a new episode.

    Instantiates a fresh ``CrisisManagementEnv`` for the requested task and
    seed, then returns the initial ``Observation`` serialised as JSON.

    Args:
        request: Contains optional ``seed`` and ``task_id`` (1–3).

    Returns:
        The initial ``Observation`` as a JSON-serialisable dict.
    """
    global _env
    try:
        _env = CrisisManagementEnv(task_id=request.task_id, seed=request.seed)
        obs: Observation = _env.reset(seed=request.seed)
        logger.info(
            "Environment reset: task_id=%d seed=%s", request.task_id, request.seed
        )
        return obs.model_dump(mode="json")
    except Exception as exc:
        logger.exception("Error during /reset: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse, tags=["openenv"])
async def step(action: Action) -> StepResponse:
    """Execute one simulation step with the provided action.

    Accepts an ``Action`` payload (JSON body) and advances the environment by
    one step.  Returns the resulting observation, scalar reward, termination
    flag, and an info dict with score / efficiency diagnostics.

    Args:
        action: The agent's dispatch decisions for this step.

    Returns:
        ``StepResponse`` containing ``observation``, ``reward``, ``done``,
        and ``info``.

    Raises:
        HTTPException 400: If the environment has not been reset yet.
        HTTPException 422: If the action payload is malformed.
        HTTPException 500: If the environment step raises any exception.
    """
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
        logger.info(
            "Step %d: reward=%.3f done=%s", obs.step, reward, done
        )
        return StepResponse(
            observation=obs.model_dump(mode="json"),
            reward=float(reward),
            done=done,
            info=info,
        )
    except Exception as exc:
        logger.exception("Error during /step: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=Dict[str, Any], tags=["openenv"])
async def state() -> Dict[str, Any]:
    """Return the current full ``EnvironmentState`` snapshot.

    This exposes more information than ``/step``'s observation — it includes
    internal counters, cumulative reward, and success flags.  Intended for
    graders and monitoring dashboards.

    Returns:
        Serialised ``EnvironmentState`` as a JSON dict.

    Raises:
        HTTPException 400: If the environment has not been reset yet.
        HTTPException 500: If state retrieval raises any exception.
    """
    env = _get_env()
    try:
        env_state: EnvironmentState = env.state()
        return env_state.model_dump(mode="json")
    except Exception as exc:
        logger.exception("Error during /state: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entrypoint (for direct execution, not needed with uvicorn CMD)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=7860, log_level="info")
