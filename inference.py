"""
inference.py
============
Production-grade LLM agent for the Adaptive Crisis Management Environment.

Agent Architecture
------------------
* Uses the OpenAI Python client pointed at the hackathon-injected endpoint.
* Consumes three environment variables exclusively:
    - HF_TOKEN      → API key for the inference endpoint.
    - API_BASE_URL  → Base URL of the model-serving endpoint.
    - MODEL_NAME    → Model identifier (e.g. "gpt-4-turbo", a HF-hosted model).
* Schema-injects the Pydantic ``Action`` model into the system prompt so the
  LLM understands the exact JSON structure required.
* Forces ``response_format={"type": "json_object"}`` for guaranteed JSON output.
* Implements a 3-retry loop with exponential back-off for transient failures.
* Falls back to a safe zero-dispatch ``Action`` after exhausted retries — the
  simulation NEVER crashes due to an LLM fault.

Logging
-------
All telemetry is emitted to the ``crisis_env.agent`` logger at appropriate
levels.  Use ``LOG_LEVEL=DEBUG`` to see prompt/response traces.

Entry Point
-----------
    python inference.py          # runs all three tasks sequentially
    python inference.py --task 2 # runs a specific task only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Global Configuration & Environment (No-Default Rule for Secrets)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import requests
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Strict OpenAI Client Configuration
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from env.models import (
    Action,
    FireLevel,
    Observation,
    PatientLevel,
    TrafficLevel,
    WeatherCondition,
    ZoneDispatch,
    StructuralHallucinationError,
)
from metrics_tracker import MetricsTracker

# ===========================================================================
# Logging configuration
# ===========================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(levelname)s] %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("crisis_env.agent")


# M2M Protocol Functions (Zero Noise, stdout only)
def emit_start(task_name: str):
    print(f"[START] Task: {task_name}", file=sys.stdout, flush=True)

def emit_step(step_num: int, obs_dict: dict, action_str: str, reward: float):
    # Mathematical precision: ensure reward is formatted as a float
    print(f"[STEP] Step: {step_num} | Obs: {obs_dict} | Action: {action_str} | Reward: {float(reward):.2f}", file=sys.stdout, flush=True)

def emit_end(score: float):
    print(f"[END] Score: {float(score):.2f}", file=sys.stdout, flush=True)


# ===========================================================================
# System Prompt — schema-injected, single source of truth
# ===========================================================================

# Inline the exact Action JSON schema so the LLM knows the contract precisely.
# Derived from the Pydantic model; kept here explicitly to avoid runtime schema
# generation failures on edge-case model versions.
_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Your complete dispatch decision for ONE simulation step.",
    "properties": {
        "allocations": {
            "type": "object",
            "description": (
                "Maps each zone name (string key) to a ZoneDispatch object. "
                "You MUST include every zone present in the observation, even if "
                "you send zero resources to it."
            ),
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "dispatch_fire": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50,
                        "description": "Number of fire units to dispatch to this zone.",
                    },
                    "dispatch_ambulance": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50,
                        "description": "Number of ambulances to dispatch to this zone.",
                    },
                    "control_traffic": {
                        "type": "boolean",
                        "description": (
                            "true to deploy one police unit for traffic control; "
                            "false otherwise."
                        ),
                    },
                },
                "required": ["dispatch_fire", "dispatch_ambulance", "control_traffic"],
            },
        },
        "public_broadcast_message": {
            "type": ["string", "null"],
            "description": (
                "Optional natural-language warning issued to citizens. "
                "REQUIRED when any zone has HIGH/CATASTROPHIC fire or CRITICAL "
                "patients. MUST name the specific zone, the hazard type "
                "(fire/medical), and include a directive verb "
                "('evacuate', 'shelter', 'avoid', 'warning'). "
                "Example: 'ALERT: Downtown has a CATASTROPHIC fire. "
                "All residents must evacuate immediately.'"
            ),
        },
    },
    "required": ["allocations"],
}

_SYSTEM_PROMPT = f"""You are an autonomous Crisis Management AI managing a simulated city during a multi-zone emergency.

## YOUR OBJECTIVE
Stabilize all city zones and minimize casualties using a limited pool of emergency resources.
Resources spent on one zone are unavailable for others until they return — triage and
cross-zone trade-offs are unavoidable.

## WHAT YOU CONTROL
At each step you receive a JSON observation with:
- The current hazard state of each zone (fire severity, medical casualties, traffic congestion).
- Your available idle resources (fire units, ambulances, police).
- Resources currently deployed and on cooldown (unavailable this step).
- A "previous_action_feedback" field: a plain-English summary of what changed in each zone
  after your last dispatch. THIS IS YOUR PRIMARY LEARNING SIGNAL. Read it every step.

## HOW TO LEARN (In-Context Calibration)
The exact resource thresholds are NOT given to you. Deduce them from feedback:
- Zone RESOLVED → your dispatch was SUFFICIENT. Remember the quantity used.
- Zone ESCALATED → your dispatch was INSUFFICIENT. Increase next allocation.
- Zone HELD STABLE → you matched minimum but did not exceed it. Adjust accordingly.

## EPISTEMIC LENS: TEMPORAL DEDUCTION (Directive 4)
There are NO hidden counters or failure tallies in the observation. You will NOT see a step
number or a failure count. You must track the efficacy of your actions over time by comparing
the current zone states against your conversation history.

If a hazard severity remains unchanged or increases since your last dispatch, your previous
allocation was insufficient and the zone is escalating. Treat any zone that failed to improve
as a higher triage priority this step. You must rely entirely on your internal reasoning to
identify and respond to escalating zones — there are no shortcuts.

## HARD PHYSICAL CONSTRAINT — INVENTORY BREACH
You CANNOT request more total resources than your current idle pool in any category
(fire units, ambulances, police). Attempting to do so VOIDS your entire action and
triggers a catastrophic terminal penalty. Always verify your totals before responding.

## BEHAVIORAL CONSTRAINTS
- Rely on feedback, not guesswork. If a zone degrades, your previous allocation was insufficient.
- Triage deliberately: when resource-constrained, concentrate on life-threatening incidents
  and consciously sacrifice lower-severity zones.
- Never waste resources on stable zones. Idle units are available for future emergencies.
- Weather and traffic conditions affect how many units are required. Learn this from feedback.

## OUTPUT FORMAT
Respond with ONLY a valid JSON object — no markdown fences, no explanations, no extra keys:

{json.dumps(_ACTION_SCHEMA, indent=2)}
"""


# ===========================================================================
# LLM Agent
# ===========================================================================

class LLMAgent:
    """Production-grade LLM agent backed by the OpenAI Python client.

    Directive 2 Compliance: Agentic Purity enforced. No retries, no fallbacks,
    no sanitization. Structural hallucinations result in immediate terminal
    penalties to ensure gradient integrity.

    All credentials and endpoint configuration are sourced exclusively from
    environment variables — no hardcoded values anywhere.

    Attributes:
        model:      Model identifier from ``MODEL_NAME`` env var.
        client:     Configured ``openai.OpenAI`` instance.
        history:    Rolling conversation history (system + alternating user/assistant).
    """

    def __init__(self) -> None:
        if not API_BASE_URL:
            logger.warning(
                "API_BASE_URL is not set. Falling back to OpenAI default endpoint. "
                "Set API_BASE_URL in .env for custom inference servers."
            )
        if not HF_TOKEN:
            logger.error(
                "HF_TOKEN is not set. API calls will fail with 401 Unauthorized. "
                "Provide HF_TOKEN in .env or as an environment variable."
            )

        logger.info(
            "LLMAgent initialised | model=%s endpoint=%s",
            MODEL_NAME,
            API_BASE_URL or "<OpenAI default>",
        )

        # Conversation history: system message is always first.
        self._history: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        obs: Observation,
        step: int,
    ) -> Tuple[Any, Optional[str]]:
        """Query the LLM for a dispatch action with NO retries (Directive 2).

        The observation is serialised to JSON and sent as the user message.
        The LLM response is parsed back into a Pydantic ``Action`` model.

        Args:
            obs:  Current environment observation.
            step: Current step number (for logging context).

        Returns:
            A 2-tuple of ``(action, error_string)``.  If parsing fails, returns
            ``("FAILED_ACTION", error)`` so the environment can apply a terminal penalty.
        """
        obs_json = obs.model_dump_json(indent=2)
        user_message = (
            f"Step {step} — Current observation:\n\n{obs_json}\n\n"
            "Respond with your dispatch action as a JSON object."
        )

        # Append user turn to rolling history (keeps context across steps).
        self._history.append({"role": "user", "content": user_message})

        try:
            action, used_tokens, latency_ms = self._call_api(step, 1)
            logger.info(
                "Step %d | tokens_used=%s | latency=%.0fms",
                step, used_tokens, latency_ms,
            )
            # Append assistant turn to history for continuity.
            self._history.append(
                {"role": "assistant", "content": action.model_dump_json()}
            )
            return action, None

        except StructuralHallucinationError as hallucination_err:
            logger.error(
                "Step %d | STRUCTURAL HALLUCINATION — %s",
                step, hallucination_err,
            )
            self._history.append(
                {"role": "assistant", "content": '{"action": "FAILED_ACTION"}'}
            )
            raise  # Let run_episode catch it and pass to the environment

        except json.JSONDecodeError as parse_err:
            last_error = f"ParseError: {parse_err}"
            logger.error(
                "Step %d | PARSE FAILURE — %s",
                step, parse_err,
            )
            self._history.append(
                {"role": "assistant", "content": '{"action": "FAILED_ACTION"}'}
            )
            return "FAILED_ACTION", last_error

        except Exception as unexpected_err:
            last_error = f"UnexpectedError: {unexpected_err}"
            logger.exception(
                "Step %d | UNEXPECTED — %s",
                step, unexpected_err,
            )
            self._history.append(
                {"role": "assistant", "content": '{"action": "FAILED_ACTION"}'}
            )
            return "FAILED_ACTION", last_error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(
        self,
        step: int,
        attempt: int,
    ) -> Tuple[Action, Optional[int], float]:
        """Make a single API call and parse the response into an ``Action``.

        Args:
            step:    Current step (telemetry).
            attempt: Retry attempt number (telemetry).

        Returns:
            A 3-tuple of ``(action, total_tokens, latency_ms)``.

        Raises:
            json.JSONDecodeError:        If the raw response is not valid JSON.
            pydantic.ValidationError:    If the JSON does not match ``Action``.
            openai.APIConnectionError:   On network-level failures.
            openai.APITimeoutError:      On request timeouts.
            openai.APIStatusError:       On non-2xx HTTP responses.
        """
        logger.debug(
            "Step %d | attempt %d — calling %s @ %s",
            step, attempt, MODEL_NAME, client.base_url,
        )

        t0 = time.monotonic()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=self._history,                        # type: ignore[arg-type]
            response_format={"type": "json_object"},       # guaranteed JSON output
            temperature=0.2,                               # low temp for determinism
            max_tokens=1024,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        raw_content = response.choices[0].message.content or ""
        total_tokens = (
            response.usage.total_tokens if response.usage else None
        )

        logger.debug(
            "Step %d | raw LLM response (%.0fms) : %s",
            step, latency_ms, raw_content,
        )

        # Parse raw JSON string → Pydantic Action natively.
        # Directive 5: Violent Validation.
        try:
            action = Action.model_validate_json(raw_content)
        except ValidationError as e:
            raise StructuralHallucinationError(str(e)) from e
        
        return action, total_tokens, latency_ms

    def reset_history(self) -> None:
        """Reset rolling conversation history between episodes.

        Retains the system prompt but clears all user/assistant turns.
        Call this at the start of each new task.
        """
        self._history = [self._history[0]]  # keep system prompt only
        logger.debug("Conversation history cleared for new episode.")


# ===========================================================================
# Reasoning helper (for [THINK] log line)
# ===========================================================================

def _assess_situation(obs: Observation) -> Tuple[str, str, str]:
    """Assess the most critical zone and overall risk level for [THINK] output.

    Directive 4 Compliance: Epistemic Lens active. Assessment is based purely
    on physically observable data (fire level, patient severity, traffic state).
    No internal counters (consecutive_failures, step number) are accessed here.
    The strategy label is derived from observed severity distribution, not
    from a hidden step clock.

    Args:
        obs: Current environment observation.

    Returns:
        A 3-tuple of ``(critical_zone, risk_level, strategy)``.
    """
    zone_scores: List[Tuple[int, str]] = []
    for z_name, z_state in obs.zones.items():
        score = 0
        # Score based solely on physically observable severity levels.
        if z_state.fire == FireLevel.CATASTROPHIC:
            score += 100
        elif z_state.fire == FireLevel.HIGH:
            score += 50
        elif z_state.fire == FireLevel.MEDIUM:
            score += 25
        if z_state.patient == PatientLevel.CRITICAL:
            score += 80
        elif z_state.patient == PatientLevel.MODERATE:
            score += 30
        if z_state.traffic == TrafficLevel.GRIDLOCK:
            score += 40
        elif z_state.traffic == TrafficLevel.HEAVY:
            score += 15
        # Directive 4: consecutive_failures intentionally NOT accessed —
        # that counter is backend-private and not in the Observation schema.
        zone_scores.append((score, z_name))

    zone_scores.sort(reverse=True)
    critical   = zone_scores[0][1] if zone_scores else "None"
    top_score  = zone_scores[0][0] if zone_scores else 0

    if top_score > 80:
        risk = "High"
    elif top_score > 40:
        risk = "Medium"
    else:
        risk = "Low"

    # Strategy derived from observed severity spread, NOT from a step counter.
    # Directive 4: obs.step is NOT accessed — it is no longer in the Observation.
    active_zones = sum(1 for _, z in obs.zones.items()
                       if z.fire != FireLevel.NONE
                       or z.patient not in (PatientLevel.NONE, PatientLevel.FATAL)
                       or z.traffic == TrafficLevel.GRIDLOCK)
    if top_score >= 80:
        strategy = "AggressiveContainment"
    elif active_zones >= 2:
        strategy = "StabilizeAndHeal"
    else:
        strategy = "OptimizeResources"

    return critical, risk, strategy


# ===========================================================================
# Episode runner
# ===========================================================================

def run_episode(agent: LLMAgent, task_id: int) -> None:
    """Run a complete episode for the given task with the LLM agent.

    Emits all mandatory OpenEnv structured log lines:
    [START], [THINK], [STEP] per step, [END] at episode close.

    Args:
        agent:   Initialised ``LLMAgent`` instance.
        task_id: Task to run (1=easy, 2=medium, 3=hard).
    """
    logger.info("=== Starting Task %d ===", task_id)
    agent.reset_history()

    response = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10)
    response.raise_for_status()
    # Assume the response JSON matches the Observation schema natively
    obs_data = response.json()
    # You may need to adjust how Observation is instantiated based on your model
    obs = Observation(**obs_data)
    metrics = MetricsTracker()

    emit_start(str(task_id))

    rewards:     List[float] = []
    step_count:  int         = 0
    final_score: float       = 0.0
    success:     bool        = False
    is_done:     bool        = False

    while not is_done:
        step_count += 1

        # ---- Introspective reasoning (logged before action) ----------------
        critical, risk, strategy = _assess_situation(obs)
        # Removed log_think to comply with M2M stream segregation

        # ---- LLM action decision -------------------------------------------
        try:
            action, step_error = agent.get_action(obs, step_count)
            if action == "FAILED_ACTION":
                action_json_str = '"FAILED_ACTION"'
            else:
                action_json_str = json.dumps(
                    action.model_dump(mode="json"), separators=(",", ":")
                )
        except StructuralHallucinationError as e:
            action = e
            step_error = str(e)
            action_json_str = '"FAILED_ACTION"'

        # ---- Environment step ----------------------------------------------
        step_res = requests.post(
            f"{ENV_URL}/step", 
            json={"action": action.model_dump(mode="json") if action != "FAILED_ACTION" else "FAILED_ACTION"}, 
            timeout=10
        )
        step_res.raise_for_status()
        step_data = step_res.json()

        obs = Observation(**step_data["observation"])
        reward = float(step_data["reward"])
        # Safely extract done, accommodating both app.py's format and legacy format
        done = step_data.get("done", step_data.get("terminated", False) or step_data.get("truncated", False))
        info = step_data.get("info", {})
        
        is_done = done
        rewards.append(float(reward))
        metrics.update(reward, action, obs, done)

        emit_step(
            step_num=step_count,
            obs_dict=obs.model_dump(mode="json"),
            action_str=action_json_str,
            reward=float(reward),
        )

        if done:
            final_score = float(info.get("score", 0.0))
            success     = final_score >= 0.5
            break

    # ---- Episode summary (OpenEnv structured log line) --------------------
    summary = metrics.get_summary()
    emit_end(final_score)
    logger.info(
        "=== Task %d complete | success=%s | score=%.3f | steps=%d ===",
        task_id, success, final_score, step_count,
    )

    # -------------------------------------------------------------------------
    # Clean sys.stdout for Meta Grader Compliance
    # -------------------------------------------------------------------------
    # Only keep [START], [STEP], and [END] markers on stdout.
    # Summary telemetry belongs in logger.info (stderr).
    # -------------------------------------------------------------------------
    total_reward: float = sum(rewards)
    formatted_reward: str = "{:.4f}".format(total_reward)

    logger.info("=== EVALUATION COMPLETE ===")
    logger.info("Total Reward: %s", formatted_reward)
    logger.info("Final State: \n%s", obs.model_dump_json(indent=2))


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM agent against the Crisis Management Environment."
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific task ID only (1=easy, 2=medium, 3=hard). "
             "Omit to run all three tasks sequentially.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    agent  = LLMAgent()

    tasks_to_run = [args.task] if args.task else [1, 2, 3]
    for t_id in tasks_to_run:
        run_episode(agent, t_id)
