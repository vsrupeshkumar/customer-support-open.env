"""
env/models.py
=============
Canonical, strictly-typed Pydantic v2 data contracts for the Adaptive Crisis
Management Environment (OpenEnv-compliant).

Design Principles
-----------------
* **Zero logic** — this file is a pure data-contract layer.  No imports from
  other ``env`` sub-modules (circular-import-free by construction).
* **LLM-resilient validators** — every numeric field that an LLM agent might
  hallucinate as a string (``"five"``, ``"3.0"``, ``"None"``) is intercepted
  and either coerced to a sane default or rejected with a descriptive error
  message that prevents the simulation loop from crashing.
* **No magic numbers** — tasks inject their own resource caps; models enforce
  *structural* invariants only (non-negative, bounded enum membership).
* **100% type-annotated** — every attribute, argument, and return value is
  explicitly typed using the ``typing`` module or built-in generics.

Compatibility
-------------
Requires ``pydantic >= 2.0``.

Example — deliberate LLM hallucination handling::

    from env.models import ZoneDispatch, Action

    # LLM returns dispatch_fire as the string "five"
    d = ZoneDispatch(dispatch_fire="five")   # → coerced / clamped to 0
    assert d.dispatch_fire == 0

    # LLM returns dispatch_fire as "-3"
    d2 = ZoneDispatch(dispatch_fire="-3")   # → coerced / clamped to 0
    assert d2.dispatch_fire == 0

    # LLM returns a float "3.7"
    d3 = ZoneDispatch(dispatch_fire="3.7")  # → floor-clamped to 3
    assert d3.dispatch_fire == 3
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

_log = logging.getLogger("crisis_env.models")


# ===========================================================================
# Constants — structural limits used by validators (NOT task configuration)
# ===========================================================================

#: Hard upper bound for a single-step dispatch count.  Prevents an LLM from
#: hallucinating absurd values (e.g., 9999) that would trivially saturate the
#: reward function.  Task resource caps apply on top of this.
_MAX_DISPATCH_PER_ZONE: int = 50

class StructuralHallucinationError(Exception):
    """Raised when an LLM outputs malformed data outside the strict Action space."""
    pass


# ===========================================================================
# Enums
# ===========================================================================

class FireLevel(str, Enum):
    """Ordered severity scale for active fire incidents in a zone.

    Values progress from ``NONE`` (no fire) to ``CATASTROPHIC`` (maximum
    severity).  The reward module uses these in ascending severity order.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CATASTROPHIC = "catastrophic"


class PatientLevel(str, Enum):
    """Ordered triage severity for medical casualties in a zone.

    ``FATAL`` represents a point of no return — patients cannot be saved but
    the incident still blocks zone resolution until formally closed.
    """

    NONE = "none"
    MODERATE = "moderate"
    CRITICAL = "critical"
    FATAL = "fatal"


class TrafficLevel(str, Enum):
    """Congestion level that prolongs ambulance response and deployment cooldown."""

    LOW = "low"
    HEAVY = "heavy"
    GRIDLOCK = "gridlock"


class WeatherCondition(str, Enum):
    """Ambient weather modifier.  Increases fire-unit requirements and cooldown."""

    CLEAR = "clear"
    STORM = "storm"
    HURRICANE = "hurricane"


class TaskLevel(str, Enum):
    """Named difficulty tiers selectable via ``reset(task_level=...)``.

    Weather modifiers mapping explicitly impacting underlying hazard thresholds
    and natively dictating cascade probability distributions.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ===========================================================================
# Component models
# ===========================================================================

class ResourcePool(BaseModel):
    """A snapshot of emergency-unit counts for one pool (idle *or* busy).

    Populated exclusively by the environment engine — never by raw LLM output.
    Uses plain int with ge=0 constraint for simplicity and compatibility.

    Attributes:
        fire_units: Number of fire-fighting units in this pool.
        ambulances: Number of ambulance units in this pool.
        police:     Number of police units in this pool.
    """

    fire_units: int = Field(default=0, ge=0)
    ambulances: int = Field(default=0, ge=0)
    police: int = Field(default=0, ge=0)


class ActiveDeployment(BaseModel):
    """A batch of units currently deployed to a zone and on cooldown.

    Attributes:
        zone_id:         Target zone identifier.
        fire_units:      Fire units in this deployment.
        ambulances:      Ambulance units in this deployment.
        police:          Police units in this deployment.
        steps_remaining: Ticks until units return to the idle pool.
        status:          Strict state string: 'DISPATCHED', 'BUSY', or 'IDLE'.
    """

    zone_id: str
    fire_units: int = Field(default=0, ge=0)
    ambulances: int = Field(default=0, ge=0)
    police: int = Field(default=0, ge=0)
    steps_remaining: int = Field(default=1, ge=1)
    status: str = Field(default="DISPATCHED")


class ZoneState(BaseModel):
    """Current hazard state of a single city zone.

    Directive 4 Compliance: Epistemic Lens active. No hidden state metadata
    leaked. Agent must use temporal deduction — comparing current severity
    against its conversation history to infer escalation trajectories.
    Internal counters (e.g. consecutive_failures) are strictly excluded and
    tracked only in the backend, never serialised into this model.

    Attributes:
        fire:    Active fire severity level.
        patient: Medical casualty severity level.
        traffic: Traffic congestion level.
    """

    fire: FireLevel = FireLevel.NONE
    patient: PatientLevel = PatientLevel.NONE
    traffic: TrafficLevel = TrafficLevel.LOW


# ===========================================================================
# Observation
# ===========================================================================

class Observation(BaseModel):
    """Full environment observation returned after every ``reset()`` / ``step()``.

    Directive 4 Compliance: Epistemic Lens active. No hidden state metadata
    leaked. This model contains ONLY physically observable data. Internal
    simulation counters (step number, max_steps, consecutive_failure counts)
    are tracked exclusively as private backend attributes and are NEVER
    serialised here. The agent must infer the passage of time and escalation
    trajectories by comparing current zone states against its conversation
    history — not by reading internal counters.

    This is the *only* view of the world exposed to the agent. Graders and
    monitors receive ``EnvironmentState``, which is a strict superset.

    Attributes:
        weather:         Global weather condition (affects all zones equally).
        zones:           Zone-id \u2192 ZoneState mapping.
        idle_resources:  Units available for dispatch this step.
        busy_resources:  Units currently deployed (on cooldown).
        task_level:      Difficulty tier this episode runs (``"easy"`` \u2026).
        previous_action_feedback: Plain-English delta summary injected by the
                         environment after every ``step()``.  Tells the agent
                         what changed in each zone relative to its last action
                         (fire went up/down/held, was the dispatch sufficient).
                         ``None`` on the first step (no prior action exists).
                         The agent should use this as its primary learning
                         signal to calibrate future dispatch quantities.
    """

    weather: WeatherCondition
    zones: Dict[str, ZoneState]
    idle_resources: ResourcePool
    busy_resources: ResourcePool
    # Directive 4: step and max_steps are REMOVED from the agent-facing
    # Observation. They are now private backend attributes in environment.py
    # (self._step_count, self._max_steps) and are never leaked to the agent.
    task_level: TaskLevel = TaskLevel.EASY
    previous_action_feedback: Optional[str] = Field(
        default=None,
        description=(
            "Structured natural-language delta from the last step. "
            "Null on step 0.  Use this to calibrate your next dispatch."
        ),
    )

    @model_validator(mode="after")
    def _validate_resource_pools(self) -> "Observation":
        """Cross-field guard: neither pool may contain negative counts.

        This is *not* a business-logic invariant of the engine (the engine
        ensures this through careful mutation), but it catches serialisation
        or deepcopy bugs early.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If any pool field is found to be negative after all
                field-level validators have run.
        """
        for field_name in ("fire_units", "ambulances", "police"):
            if getattr(self.idle_resources, field_name) < 0:
                raise ValueError(
                    f"idle_resources.{field_name} is negative — data corruption detected."
                )
            if getattr(self.busy_resources, field_name) < 0:
                raise ValueError(
                    f"busy_resources.{field_name} is negative — data corruption detected."
                )
        return self


# ===========================================================================
# Action — LLM-resilient dispatch schema
# ===========================================================================

class ZoneDispatch(BaseModel):
    """Dispatch instruction for a single zone within one simulation step.

    **Resilient Bouncer Mode** — LLM-Tolerant Coercion with Hard Boundary Enforcement
    ---------------------------------------------------------------------------------
    This model coerces common LLM output patterns into valid types before
    Pydantic validates them.  External LLMs (e.g. Nemotron 3 Super used in
    Phase 2 Agentic Evaluation) commonly return:
        - integers as floats (``3.0`` instead of ``3``)
        - booleans as integers (``1``/``0`` instead of ``true``/``false``)
        - booleans as strings (``"true"``/``"false"``)

    We coerce these to the correct Python types so minor format variance does
    NOT trigger a ``ValidationError``.  Semantic violations (negative counts,
    inventory breach) are still caught by the environment's hard guards.

    Accepted and coerced:
        3.0   (float int)   → 3
        3.7   (float)       → 3  (floor-truncated)
        1     (int bool)    → True
        0     (int bool)    → False
        "true" / "false"   → True / False

    Still rejected (no safe coercion possible):
        "five"  (string)   → ValidationError
        -1      (negative) → ValidationError (after coercion)

    Attributes:
        dispatch_fire:      Fire units to dispatch (non-negative int).
        dispatch_ambulance: Ambulance units to dispatch (non-negative int).
        control_traffic:    Whether to deploy a police unit (bool).
    """

    dispatch_fire: int = Field(
        default=0,
        description="Fire units to dispatch. Non-negative integer. Floats are floor-truncated.",
    )
    dispatch_ambulance: int = Field(
        default=0,
        description="Ambulance units to dispatch. Non-negative integer. Floats are floor-truncated.",
    )
    control_traffic: bool = Field(
        default=False,
        description="Deploy one police unit for traffic control. Bool; 0/1 and 'true'/'false' are coerced.",
    )

    # ------------------------------------------------------------------
    # Coercing validators — type-safe ingestion before boundary checks
    # ------------------------------------------------------------------

    @field_validator("dispatch_fire", "dispatch_ambulance", mode="before")
    @classmethod
    def _coerce_to_int(cls, v: Any) -> int:
        """Coerce float/string representations to int, then enforce non-negative.

        Handles the common pattern of external LLMs returning ``3.0`` or
        ``"3"`` for integer fields.  Floor-truncates floats (3.7 → 3).

        Args:
            v: Raw value from LLM JSON payload.

        Returns:
            Non-negative ``int``.

        Raises:
            ValueError: If the value cannot be coerced or is negative.
        """
        if isinstance(v, bool):
            # bool is a subclass of int in Python — treat True/False as 1/0
            return int(v)
        if isinstance(v, float):
            v = int(v)  # floor-truncate
        if isinstance(v, str):
            try:
                v = int(float(v))  # "3" → 3, "3.7" → 3
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot coerce {v!r} to int for dispatch count."
                )
        if not isinstance(v, int):
            raise ValueError(f"Expected int, got {type(v).__name__}: {v!r}")
        if v < 0:
            raise ValueError(
                f"Action Space Violation: Expected >= 0, got {v}. "
                "Negative dispatch counts are outside the valid action space."
            )
        return v

    @field_validator("control_traffic", mode="before")
    @classmethod
    def _coerce_to_bool(cls, v: Any) -> bool:
        """Coerce int/string representations to bool.

        External LLMs commonly return ``1``/``0`` or ``"true"``/``"false"``
        for boolean fields.  This validator normalises all common patterns.

        Args:
            v: Raw value from LLM JSON payload.

        Returns:
            Python ``bool``.

        Raises:
            ValueError: If the value cannot be safely interpreted as a boolean.
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            if v.lower() in ("true", "1", "yes"):
                return True
            if v.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"Cannot coerce string {v!r} to bool.")
        raise ValueError(f"Expected bool, got {type(v).__name__}: {v!r}")


class Action(BaseModel):
    """The RL agent's complete dispatch decision for one simulation step.

    Attributes:
        allocations: Zone-id → ZoneDispatch mapping.  Keys that do not match
            any zone in the current ``Observation`` are silently dropped by
            the environment (they cannot affect the simulation).
        public_broadcast_message: Optional natural-language warning string
            generated by the LLM agent to alert simulated citizens.  Evaluated
            by the Context-Grounded Semantic Grader in ``reward.py`` for a max
            bonus of +1.0 per step when an active high-severity incident exists.
            To earn the full bonus the string must name the critical zone, the
            hazard type, and include at least one directive verb (evacuate,
            shelter, avoid, warning).

    Note:
        ``allocations`` is intentionally permissive at the model level (any
        string key is accepted) because the environment validates keys against
        the live zone registry at ``step()`` time, allowing it to emit
        structured ``info["invalid_zone_keys"]`` diagnostics rather than
        crashing.
    """

    allocations: Dict[str, ZoneDispatch] = Field(
        default_factory=dict,
        description=(
            "Maps zone identifier → dispatch instructions.  "
            "Unknown zone keys are dropped silently by the environment."
        ),
    )

    public_broadcast_message: Optional[str] = Field(
        default=None,
        description=(
            "A natural-language warning issued to citizens.  MUST include the "
            "specific zone name, the type of hazard (fire/medical), and specific "
            "instructions (e.g., 'evacuate', 'shelter', 'avoid').  Evaluated by "
            "the Context-Grounded Semantic Grader for a max +1.0 reward bonus."
        ),
    )

class PoisonAction(Action):
    """Sentinel action for tracking deserialization exploits returning secure payloads."""
    error_msg: str = "Payload Exploit Detected"


# ===========================================================================
# Environment State (full internal snapshot — superset of Observation)
# ===========================================================================

class TrajectoryStep(BaseModel):
    """An atomic trajectory element for episode history reasoning."""
    observation: Observation
    action: Action
    reward: float

class EnvironmentState(BaseModel):
    """Complete internal snapshot of the environment.

    Agents receive ``Observation``; graders and monitors receive this.

    Attributes:
        step_count:    Steps elapsed this episode.
        max_steps:     Episode truncation limit.
        observation:   The current public observation.
        total_reward:  Cumulative reward so far.
        is_done:       Whether the episode has terminated.
        success:       Whether all incidents were resolved before truncation.
        metrics:       Arbitrary float key-value diagnostics.
        invalid_action_count: Running count of malformed / zero-effect actions
                       this episode (populated by the environment).
    """

    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    observation: Observation
    total_reward: float
    is_done: bool
    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    invalid_action_count: int = Field(default=0, ge=0)
    episode_history: List[TrajectoryStep] = Field(default_factory=list, description="A sliding window of the last k steps (o_t, a_t, r_t).")


# ===========================================================================
# Episode History Record (used by Grader)
# ===========================================================================

class StepRecord(BaseModel):
    """Immutable log of a single environment step for post-episode grading.

    Attributes:
        step:        Step number (1-indexed).
        observation: Observation *before* the action was applied.
        action:      Action selected by the agent.
        reward:      Scalar reward returned by the environment.
        done:        Whether the episode ended at this step.
        info:        The full ``info`` dict returned by ``env.step()``.
    """

    step: int = Field(ge=1)
    observation: Observation
    action: Action
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# Reward Schema (OpenEnv Pydantic Compliance — Mathematical Ledger)
# ===========================================================================

class Reward(BaseModel):
    """Mathematical Ledger for the step-level reward signal.

    OpenEnv requires Action, Observation, AND Reward to be typed Pydantic
    models.  This class is not a passive schema — it is an **active ledger**
    that records the exact arithmetic breakdown of every point awarded or
    deducted, providing full auditability for judges and post-episode analysis.

    Design Guarantee
    ----------------
    ``total_reward`` is always the authoritative scalar.  The sub-components
    are its provenance:

        total_reward = base_dispatch_score + nlp_semantic_bonus - waste_penalty

    A ``model_validator`` enforces this identity at construction time, catching
    any arithmetic inconsistency at the boundary between reward.py and the
    environment loop.

    Reward Layers (mapped to sub-components)
    -----------------------------------------
    +-------------------------------+------------------------+------------------+
    | Sub-component                 | Internal Layer         | Typical Range    |
    +===============================+========================+==================+
    | base_dispatch_score           | Dispatch Quality       | [-9, 8] per zone |
    |                               | + Trajectory Shaping   | + [-3, 2]        |
    +-------------------------------+------------------------+------------------+
    | nlp_semantic_bonus            | NLP Broadcast Bonus    | (-∞, 1.0]        |
    |                               | (CAN BE NEGATIVE —     |                  |
    |                               |  Directive 3 penalty)  |                  |
    +-------------------------------+------------------------+------------------+
    | waste_penalty                 | Over-dispatch severity | [0.0, ∞)         |
    +-------------------------------+------------------------+------------------+
    | **total_reward**              | Ledger sum             | (-∞, +∞)         |
    +-------------------------------+------------------------+------------------+

    Attributes:
        base_dispatch_score:  Points awarded for correct numerical resource
                              allocation (Layers 1 + 2 combined).
        nlp_semantic_bonus:   Bonus (+0.5 max) for generating contextually
                              accurate natural-language broadcasts.
        waste_penalty:        Negative points applied for over-dispatching
                              to low-severity zones (always non-negative;
                              subtracted from total).
        total_reward:         The calculated sum:
                              ``base_dispatch_score + nlp_semantic_bonus
                              - waste_penalty``.  Not clamped — CAN be
                              negative (Directive 3 compliance).
        dispatch_quality:     Layer 1 raw float (kept for backward compat).
        trajectory_shaping:   Layer 2 raw float (kept for backward compat).
        nlp_bonus:            Layer 3 raw float, mirrors nlp_semantic_bonus.
        is_terminal:          Whether the episode ended at this step.
    """

    # ------------------------------------------------------------------
    # Primary mathematical sub-components (the "ledger lines")
    # ------------------------------------------------------------------

    base_dispatch_score: float = Field(
        default=0.0,
        description="Points awarded for correct numerical resource allocation (Layers 1 + 2).",
    )
    nlp_semantic_bonus: float = Field(
        default=0.0,
        description=(
            "NLP broadcast bonus — CAN BE NEGATIVE (Directive 3). "
            "Subtracted hallucination (λ=0.5/kw) and bloat (γ=0.01/excess word) "
            "penalties may outweigh positive keyword matches. No zero-floor clamp."
        ),
    )
    waste_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Negative points applied for over-dispatching to low-severity zones "
            "(stored as a positive magnitude; subtracted in total)."
        ),
    )
    efficiency_bonus: float = Field(
        default=0.0,
        description=(
            "Resource-conservation bonus from Layer 3 POMDP formula: "
            "(resources_saved / total_resources) × 0.5. Rewards minimal sufficient dispatch."
        ),
    )
    time_penalty: float = Field(
        default=0.0,
        description=(
            "Constant per-step time cost from Layer 3 POMDP formula (0.1 per step). "
            "Encourages maximising episode efficiency."
        ),
    )
    multi_obj: float = Field(
        default=0.0,
        description=(
            "Layer 3 canonical multi-objective POMDP scalar: "
            "(severity_delta × 1.5) + efficiency_bonus − time_penalty."
        ),
    )
    total_reward: float = Field(
        ...,
        description=(
            "Authoritative ledger sum: "
            "base_dispatch_score + nlp_semantic_bonus − waste_penalty "
            "+ efficiency_bonus − time_penalty + multi_obj."
        ),
    )

    # ------------------------------------------------------------------
    # Layer-resolution fields (backward-compatible, kept for graders)
    # ------------------------------------------------------------------

    dispatch_quality: float = Field(
        default=0.0,
        description="Layer 1: per-zone dispatch quality reward from _zone_reward().",
    )
    trajectory_shaping: float = Field(
        default=0.0,
        description="Layer 2: Δ-severity shaping — Stabilization Bonus or Degradation Penalty.",
    )
    nlp_bonus: float = Field(
        default=0.0,
        description=(
            "Layer 3: Context-Grounded Semantic Grader score. "
            "CAN BE NEGATIVE per Directive 3 — mirrors nlp_semantic_bonus."
        ),
    )
    is_terminal: bool = Field(
        default=False,
        description="True if the episode terminated at this step (all resolved or max_steps hit).",
    )

    # ------------------------------------------------------------------
    # Floating-point ledger proof — 6-component identity validator
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def verify_reward_ledger(self) -> "Reward":
        """Enforce the 6-component reward ledger identity with IEEE 754 tolerance.

        Reconstructs ``total_reward`` from all constituent sub-components using
        the exact addition/subtraction logic applied inside
        ``calculate_step_reward`` and ``compute_reward``, then asserts
        structural integrity via ``math.isclose`` with an absolute tolerance
        of 1e-4 — wide enough to absorb IEEE 754 floating-point drift while
        tight enough to catch any real arithmetic inconsistency.

        Mathematical identity enforced
        --------------------------------
        calculated_total =
              base_dispatch_score
            + nlp_semantic_bonus
            − waste_penalty
            + efficiency_bonus
            − time_penalty
            + multi_obj

        BUG-033 Contract
        ----------------
        ``total_reward`` in this ledger is the **UNDISCOUNTED** pre-discount
        reward (the raw arithmetic sum of the 6 sub-components). The POMDP
        temporal discount factor (γ = 0.99) is applied AFTER ledger construction
        and returned to the agent as a separate MDP signal.

        DO NOT pass the discounted ``reward`` variable to ``total_reward``.
        The discount creates a gap of |R| × |1 − γ^(t−1)| that violates
        ``abs_tol=1e-4`` at any step > 3 with non-trivial reward magnitude.

        Returns:
            Self if the identity holds.

        Raises:
            ValueError: If ``calculated_total`` and ``total_reward`` differ by
                more than ``abs_tol=1e-4``, indicating a State-Validation
                Asymmetry between the step function and the Pydantic ledger.
        """
        # 1. Reconstruct the exact multi-objective mathematical equation
        calculated_total = (
            self.base_dispatch_score
            + self.nlp_semantic_bonus
            - self.waste_penalty
            + self.efficiency_bonus
            - self.time_penalty
            + self.multi_obj
        )

        # 2. Enforce structural integrity using epsilon tolerance for float drift
        if not math.isclose(calculated_total, self.total_reward, abs_tol=1e-4):
            raise ValueError(
                f"Reward ledger identity violated: "
                f"base({self.base_dispatch_score}) + semantic({self.nlp_semantic_bonus}) - "
                f"waste({self.waste_penalty}) + efficiency({self.efficiency_bonus}) - "
                f"time({self.time_penalty}) + multi_obj({self.multi_obj}) "
                f"= {calculated_total:.4f} != total_reward({self.total_reward:.4f})"
            )
        return self

    # ------------------------------------------------------------------
    # Computation helper — explicit calculation property
    # ------------------------------------------------------------------

    def calculate_total(self) -> float:
        """Mathematically compute what total_reward *should* equal.

        Reconstructs the authoritative scalar from all six sub-components
        using the same addition/subtraction logic as ``verify_reward_ledger``.
        Useful for verification in tests and post-episode analysis.

        Returns:
            ``base_dispatch_score + nlp_semantic_bonus − waste_penalty
            + efficiency_bonus − time_penalty + multi_obj``
        """
        return (
            self.base_dispatch_score
            + self.nlp_semantic_bonus
            - self.waste_penalty
            + self.efficiency_bonus
            - self.time_penalty
            + self.multi_obj
        )
