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

    These map deterministically to a ``TaskConfig`` object inside
    ``env.tasks``.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ===========================================================================
# Task Configuration (no magic numbers in the environment loop)
# ===========================================================================

class TaskConfig(BaseModel):
    """Fully parameterised configuration for one task difficulty tier.

    All values that would otherwise be hardcoded magic numbers in the
    environment loop are extracted here.  The environment reads these at
    ``reset()`` time.

    Attributes:
        task_id:                 Numeric identifier (1, 2, 3 …).
        name:                   Human-readable difficulty label.
        max_steps:               Episode step limit before truncation.
        cascade_threshold:       Consecutive failures required to trigger
                                 incident escalation.
        cooldown_base:           Default deployment cooldown in steps.
        cooldown_storm_modifier: Additional cooldown steps during STORM.
        cooldown_hurricane_modifier: Additional cooldown steps during HURRICANE.
        cooldown_gridlock_modifier: Additional cooldown steps when zone has
                                 GRIDLOCK traffic.
        lives_saved_per_amb:     Lives credited per successful ambulance
                                 resolution (used by metrics, not reward).
    """

    task_id: int = Field(ge=1)
    name: str
    max_steps: int = Field(ge=1)
    cascade_threshold: int = Field(default=3, ge=1)
    cooldown_base: int = Field(default=1, ge=1)
    cooldown_storm_modifier: int = Field(default=1, ge=0)
    cooldown_hurricane_modifier: int = Field(default=2, ge=0)
    cooldown_gridlock_modifier: int = Field(default=2, ge=0)
    lives_saved_per_amb: int = Field(default=18, ge=0)


# ===========================================================================
# Component models
# ===========================================================================

class ResourcePool(BaseModel):
    """A snapshot of emergency-unit counts for one pool (idle *or* busy).

    Attributes:
        fire_units: Number of fire-fighting units in this pool.
        ambulances: Number of ambulance units in this pool.
        police:     Number of police units in this pool.
    """

    fire_units: int = Field(default=0, ge=0)
    ambulances: int = Field(default=0, ge=0)
    police: int = Field(default=0, ge=0)

    @field_validator("fire_units", "ambulances", "police", mode="before")
    @classmethod
    def _parse_resource_count(cls, raw: Any) -> int:
        """Coerce and validate a resource count.

        Accepts integers, floats (floor-truncated), and numeric strings.
        Rejects non-numeric strings.  Negative values are clamped to 0.

        Args:
            raw: The raw value from whichever caller populated the field.

        Returns:
            A non-negative integer.

        Raises:
            ValueError: If ``raw`` cannot be interpreted as a number.
        """
        if isinstance(raw, bool):
            # bool is a subclass of int in Python; reject it explicitly.
            raise ValueError(
                f"Resource count must be an integer, got boolean {raw!r}."
            )
        try:
            coerced = int(float(str(raw)))
        except (ValueError, TypeError):
            raise ValueError(
                f"Resource count must be a non-negative integer, got {raw!r}."
            )
        if coerced < 0:
            _log.debug("Resource count %d clamped to 0.", coerced)
            return 0
        return coerced


class ActiveDeployment(BaseModel):
    """A batch of units currently deployed to a zone and on cooldown.

    Attributes:
        zone_id:         Target zone identifier.
        fire_units:      Fire units in this deployment.
        ambulances:      Ambulance units in this deployment.
        police:          Police units in this deployment.
        steps_remaining: Ticks until units return to the idle pool.
    """

    zone_id: str
    fire_units: int = Field(default=0, ge=0)
    ambulances: int = Field(default=0, ge=0)
    police: int = Field(default=0, ge=0)
    steps_remaining: int = Field(default=1, ge=1)


class ZoneState(BaseModel):
    """Current hazard state of a single city zone.

    Attributes:
        fire:                Active fire severity level.
        patient:             Medical casualty severity level.
        traffic:             Traffic congestion level.
        consecutive_failures: Steps with insufficient response before cascade.
    """

    fire: FireLevel = FireLevel.NONE
    patient: PatientLevel = PatientLevel.NONE
    traffic: TrafficLevel = TrafficLevel.LOW
    consecutive_failures: int = Field(default=0, ge=0)


# ===========================================================================
# Observation
# ===========================================================================

class Observation(BaseModel):
    """Full environment observation returned after every ``reset()`` / ``step()``.

    This is the *only* view of the world exposed to the agent.  Graders and
    monitors receive ``EnvironmentState``, which is a strict superset.

    Attributes:
        weather:         Global weather condition (affects all zones equally).
        zones:           Zone-id → ZoneState mapping.
        idle_resources:  Units available for dispatch this step.
        busy_resources:  Units currently deployed (on cooldown).
        step:            Current step index (0 at episode start).
        max_steps:       Truncation limit for this episode.
        task_level:      Difficulty tier this episode runs (``"easy"`` …).
    """

    weather: WeatherCondition
    zones: Dict[str, ZoneState]
    idle_resources: ResourcePool
    busy_resources: ResourcePool
    step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=10, ge=1)
    task_level: TaskLevel = TaskLevel.EASY

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

    This model is the primary attack-surface for erratic LLM outputs.  Every
    numeric field uses a permissive ``mode="before"`` validator that converts
    strings, floats, and ``None`` into safe integer values rather than crashing.

    Attributes:
        dispatch_fire:      Fire units to send (non-negative, capped).
        dispatch_ambulance: Ambulance units to send (non-negative, capped).
        control_traffic:    Whether to deploy a police unit.

    Examples:
        >>> ZoneDispatch(dispatch_fire="five")          # → 0 (unparsable → 0)
        >>> ZoneDispatch(dispatch_fire="3.7")           # → 3 (floor truncation)
        >>> ZoneDispatch(dispatch_fire="-3")            # → 0 (negative → clamp)
        >>> ZoneDispatch(dispatch_fire=None)            # → 0 (None → 0)
        >>> ZoneDispatch(control_traffic="yes")         # → True
        >>> ZoneDispatch(control_traffic="false")       # → False
        >>> ZoneDispatch(control_traffic=0)             # → False
    """

    dispatch_fire: int = Field(
        default=0,
        ge=0,
        le=_MAX_DISPATCH_PER_ZONE,
        description="Fire units to dispatch (0–50).  Clamped to idle pool at simulation time.",
    )
    dispatch_ambulance: int = Field(
        default=0,
        ge=0,
        le=_MAX_DISPATCH_PER_ZONE,
        description="Ambulance units to dispatch (0–50).  Clamped to idle pool at simulation time.",
    )
    control_traffic: bool = Field(
        default=False,
        description="Deploy one police unit for traffic control.",
    )

    # ------------------------------------------------------------------
    # Numeric field validators (LLM hallucination safety net)
    # ------------------------------------------------------------------

    @field_validator("dispatch_fire", "dispatch_ambulance", mode="before")
    @classmethod
    def _coerce_dispatch_count(cls, raw: Any) -> int:
        """Convert an LLM-generated value to a safe non-negative integer.

        Behaviour by input type:

        +-----------------------+---------------------------+
        | Input                 | Output                    |
        +=======================+===========================+
        | ``5`` (int)           | ``5``                     |
        +-----------------------+---------------------------+
        | ``3.7`` (float)       | ``3`` (floor-truncated)   |
        +-----------------------+---------------------------+
        | ``"3"`` (str int)     | ``3``                     |
        +-----------------------+---------------------------+
        | ``"3.7"`` (str float) | ``3`` (floor-truncated)   |
        +-----------------------+---------------------------+
        | ``"-3"`` (negative)   | ``0`` (clamped)           |
        +-----------------------+---------------------------+
        | ``"five"`` (word)     | ``0`` (unparsable)        |
        +-----------------------+---------------------------+
        | ``None``              | ``0`` (missing → zero)    |
        +-----------------------+---------------------------+
        | ``True`` / ``False``  | ``ValueError`` (rejected) |
        +-----------------------+---------------------------+

        Args:
            raw: The raw value supplied by the agent or deserialiser.

        Returns:
            A non-negative integer safe to use for dispatch.

        Raises:
            ValueError: Only when ``raw`` is a boolean (ambiguous intent).
        """
        if isinstance(raw, bool):
            raise ValueError(
                f"dispatch count must be an integer, not a boolean ({raw!r}). "
                "Use 0 or 1 instead."
            )
        if raw is None:
            _log.debug("dispatch count was None; defaulting to 0.")
            return 0
        try:
            coerced = int(float(str(raw)))
        except (ValueError, TypeError):
            _log.warning(
                "Unparsable dispatch value %r from agent; defaulting to 0.", raw
            )
            return 0
        if coerced < 0:
            _log.debug("Negative dispatch %d clamped to 0.", coerced)
            return 0
        return min(coerced, _MAX_DISPATCH_PER_ZONE)

    # ------------------------------------------------------------------
    # Boolean field validator (LLM often returns "yes"/"no"/"1"/"0")
    # ------------------------------------------------------------------

    @field_validator("control_traffic", mode="before")
    @classmethod
    def _coerce_bool(cls, raw: Any) -> bool:
        """Convert permissive truthy values to a strict Python ``bool``.

        Handles LLM outputs such as ``"yes"``, ``"true"``, ``"1"``,
        ``"no"``, ``"false"``, ``"0"``, ``None``, or integers.

        Args:
            raw: Raw value from the agent or deserialiser.

        Returns:
            ``True`` or ``False``.
        """
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int):
            return raw != 0
        if isinstance(raw, str):
            normalised = raw.strip().lower()
            if normalised in {"true", "yes", "1", "on"}:
                return True
            if normalised in {"false", "no", "0", "off", "none", "null"}:
                return False
            _log.warning(
                "Unrecognised bool string %r for control_traffic; defaulting to False.",
                raw,
            )
            return False
        if raw is None:
            return False
        # Fall-through: trust Python truthiness.
        return bool(raw)


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

    @field_validator("allocations", mode="before")
    @classmethod
    def _coerce_allocations(cls, raw: Any) -> Dict[str, Any]:
        """Ensure ``allocations`` is always a dict, even if LLM omits it.

        Args:
            raw: Raw allocations payload.

        Returns:
            A dict (possibly empty).

        Raises:
            ValueError: If ``raw`` is not dict-like and cannot be defaulted.
        """
        if raw is None:
            _log.warning("Action.allocations was None; defaulting to empty dict.")
            return {}
        if isinstance(raw, dict):
            return raw
        _log.warning(
            "Action.allocations expected dict, got %s; defaulting to empty dict.",
            type(raw).__name__,
        )
        return {}


# ===========================================================================
# Environment State (full internal snapshot — superset of Observation)
# ===========================================================================

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
    | nlp_semantic_bonus            | NLP Broadcast Bonus    | [0.0, 1.0]       |
    +-------------------------------+------------------------+------------------+
    | waste_penalty                 | Over-dispatch severity | [0.0, ∞)         |
    +-------------------------------+------------------------+------------------+
    | **total_reward**              | Ledger sum             | ≈ -12 .. +11     |
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
                              - waste_penalty``.
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
        description="Points awarded for correct numerical resource allocation.",
    )
    nlp_semantic_bonus: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Bonus (+0.5 max) awarded for generating contextually accurate natural language broadcasts.",
    )
    waste_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description="Negative points applied for over-dispatching to low-severity zones (stored as a positive magnitude; subtracted in total).",
    )
    total_reward: float = Field(
        ...,
        description="The calculated sum: base_dispatch_score + nlp_semantic_bonus - waste_penalty.",
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
        ge=0.0,
        le=1.0,
        description="Layer 3: Context-Grounded Semantic Grader score (0.0–1.0). Mirrors nlp_semantic_bonus.",
    )
    is_terminal: bool = Field(
        default=False,
        description="True if the episode terminated at this step (all resolved or max_steps hit).",
    )

    # ------------------------------------------------------------------
    # Mathematical guarantee — ledger identity validator
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _assert_ledger_identity(self) -> "Reward":
        """Assert that total_reward equals the declared ledger sum.

        Allows a floating-point tolerance of 1e-9 to absorb IEEE-754 rounding.

        Returns:
            Self if the identity holds.

        Raises:
            ValueError: If the arithmetic identity is violated.
        """
        expected = self.base_dispatch_score + self.nlp_semantic_bonus - self.waste_penalty
        if abs(self.total_reward - expected) > 1e-9:
            raise ValueError(
                f"Reward ledger identity violated: "
                f"base_dispatch_score({self.base_dispatch_score:.4f}) "
                f"+ nlp_semantic_bonus({self.nlp_semantic_bonus:.4f}) "
                f"- waste_penalty({self.waste_penalty:.4f}) "
                f"= {expected:.4f} ≠ total_reward({self.total_reward:.4f})."
            )
        return self

    # ------------------------------------------------------------------
    # Computation helper — explicit calculation property
    # ------------------------------------------------------------------

    def calculate_total(self) -> float:
        """Mathematically compute what total_reward *should* equal.

        This method provides an explicit calculation of the reward sum
        independent of the stored ``total_reward`` field, useful for
        verification in tests and post-episode analysis.

        Returns:
            ``base_dispatch_score + nlp_semantic_bonus - waste_penalty``
        """
        return self.base_dispatch_score + self.nlp_semantic_bonus - self.waste_penalty
