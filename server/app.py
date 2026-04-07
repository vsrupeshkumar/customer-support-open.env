"""
server.py
=========
Minimal FastAPI wrapper that exposes the CrisisManagementEnv as a
Hugging Face-compatible HTTP API service.
"""

from __future__ import annotations

import logging
import os
import json
import math
import random
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
from env.models import Action, EnvironmentState, Observation, StructuralHallucinationError

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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    global _env
    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "Environment not initialised. Call POST /reset first."}
        )
    
    # Mathematical Penalty Calculus
    _env._step_count += 1
    reward = -20.0
    done = _env._step_count >= _env._max_steps
    
    if done:
        _env._is_done = True
        
    info = {
        "error_type": "RequestValidationError",
        "detail": "LLM generated an invalid JSON schema."
    }
    
    # Force the 200 OK with the exact StepResponse schema
    step_resp = StepResponse(
        observation=_env.obs.model_dump(mode="json"),
        reward=reward,
        done=done,
        info=info
    )
    
    return JSONResponse(status_code=200, content=step_resp.model_dump())

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
async def reset(request: Request) -> Dict[str, Any]:
    global _env
    try:
        data = await request.json()
        task_id = int(data.get("task_id", 1))
        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)
        else:
            seed = random.randint(1, 100000)
    except Exception:
        # Fallback if the body is missing or malformed to avoid 422 errors
        task_id = 1
        seed = random.randint(1, 100000)

    try:
        _env = CrisisManagementEnv(task_id=task_id, seed=seed)
        obs = _env.obs  # Access the cleanly initialized state instead of double-calling reset
        logger.info("Environment initialized: task_id=%d seed=%s", task_id, seed)

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
        obs_dict["Environment_Complexity"] = round(entropy, 4)

        # Log EVENT START
        log_event("START", {"task": task_id, "seed": seed, "entropy": round(entropy, 4)})
        
        return obs_dict
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
    """
    env = _get_env()
    try:
        data = await request.json()
        action = Action(**data)
    except Exception as e:
        action = StructuralHallucinationError(str(e))

    try:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        logger.info("step reward=%.4f done=%s", reward, done)

        if done:
            success = info.get("resolved", 0) == info.get("total", 0)
            log_event("END", {"success": str(success).lower(), "score": info.get("score", 0.0)})

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
        safe_obs = _env.obs.model_dump(mode="json") if _env is not None else {}
        return StepResponse(
            observation=safe_obs,
            reward=-5.0,
            done=False,
            info={"error": f"Internal environment error: {exc}"},
        )

@app.get("/state", response_model=Dict[str, Any], tags=["openenv"])
async def state() -> Dict[str, Any]:
    env = _get_env()
    try:
        env_state: EnvironmentState = env.state
        return env_state.model_dump(mode="json")
    except Exception as exc:
        logger.exception("Error during /state: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
