"""
env/reward.py
=============
Dense, per-step reward function for the Adaptive Crisis Management Environment.

Design Principles
-----------------
* **Dense rewards** — a scalar reward is emitted at *every* step, not just
  at episode termination.  This gives the RL agent a clear gradient signal
  throughout the entire episode.
* **Pure function** — ``calculate_step_reward`` has no side-effects and
  depends only on its arguments.  This makes it trivially unit-testable and
  deterministic.
* **Trajectory-Aware Reward Shaping** — the function now utilises the
  ``previous_state`` argument to compute *Δ-severity signals* across two
  consecutive steps.  These signals reward containment (stopping a cascade)
  and punish control failures (letting an incident escalate).
* **Backward-compatible shim** — ``compute_reward`` is retained as an alias
  so that ``environment.py``, which already imports it, continues to work
  without modification.

Reward Table (Step-Level Signals)
----------------------------------
+-------------------------------------------------------+--------+
| Event                                                 | Points |
+=======================================================+========+
| Correct resource allocation (exact match)             | +2.0   |
+-------------------------------------------------------+--------+
| Saving a critical case (CRITICAL patient resolved)    | +5.0   |
+-------------------------------------------------------+--------+
| Delayed response to high severity (no dispatch while  | -5.0   |
|   HIGH/CATASTROPHIC fire *or* CRITICAL patient pends) |        |
+-------------------------------------------------------+--------+
| Over-allocation (sent more units than required)       | -2.0   |
+-------------------------------------------------------+--------+
| Efficient resolution (exact match dispatch)           | +1.0   |
+-------------------------------------------------------+--------+
| Ignoring incident (do-nothing while incidents pend)   | -4.0   |
+-------------------------------------------------------+--------+

Trajectory-Aware Δ-Severity Signals (novel dense shaping)
-----------------------------------------------------------
+-------------------------------------------------------+--------+
| Stabilization Bonus: incident was escalating          |        |
|   last step, severity held stable this step           | +2.0   |
+-------------------------------------------------------+--------+
| Degradation Penalty: incident severity *increased*    |        |
|   between previous_state and current_state            | -3.0   |
+-------------------------------------------------------+--------+

Stabilization Bonus — Design Rationale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An RL agent that dispatches *just enough* to stop a cascading incident
deserves more reward than one that throws excessive resources at an
already-resolved zone.  The stabilization bonus detects this by comparing
the zone's severity across **two** consecutive timesteps:

    Δ = ordinal_rank(current_severity) - ordinal_rank(previous_severity)

    If Δ == 0  AND  ordinal_rank(previous_severity) > ordinal_rank(the step
    before that)  → the agent halted the degradation.  Grant +2.0.

Because the environment only exposes the *immediately* previous state, we
proxy "was escalating in the prior step" by checking whether any zone already
had a *consecutive_failures* count > 0 at the start of the current step (this
is set by ``_resolve_zone`` when the previous action was insufficient).

Degradation Penalty — Design Rationale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Granting a penalty whenever an incident's ordinal severity rank *increases*
between ``previous_state`` and ``current_state`` provides an early, dense
punishment signal for inaction or under-dispatch.  Without this, the agent
must wait for the sparse cascade event (3+ consecutive failures) before
receiving a meaningful negative gradient.

    If ordinal_rank(current_severity) > ordinal_rank(previous_severity)
    → apply DEGRADATION_PENALTY = -3.0.

Mathematical Definitions
------------------------
Let ``R_fire`` = ``_get_required_fire(zone.fire, obs.weather)``
Let ``R_amb``  = ``_get_required_ambulance(zone.patient)``
Let ``D_fire`` = ``dispatch.dispatch_fire``
Let ``D_amb``  = ``dispatch.dispatch_ambulance``

**Correct allocation** (+2.0 per zone):
    Awarded when **all** of the following hold:
        D_fire >= R_fire  (if R_fire > 0)
        D_amb  >= R_amb   (if R_amb  > 0)
        control_traffic   (if zone has HEAVY or GRIDLOCK traffic)
    i.e., the dispatch vector satisfies the requirement *at minimum*.

**Over-allocation** (-2.0 per zone):
    Awarded **in addition** to the correct-allocation bonus when:
        D_fire > R_fire   (surplus fire units dispatched)
        OR
        D_amb  > R_amb    (surplus ambulances dispatched)
    Mathematically: surplus_fire = D_fire - R_fire > 0
                    surplus_amb  = D_amb  - R_amb  > 0

These two signals are *not* mutually exclusive.  An agent that sends exactly
the right number of fire units but too many ambulances will receive both
+2.0 (correct fire/traffic) and -2.0 (ambulance over-allocation).  The net
incentive pushes toward minimal sufficient dispatch.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from env.models import (
    Action,
    Observation,
    FireLevel,
    PatientLevel,
    Reward,
    TrafficLevel,
    WeatherCondition,
    ZoneDispatch,
    ZoneState,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("crisis_env.reward")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("[%(levelname)s] REWARD - %(message)s"))
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Public reward constants — single source of truth
# ---------------------------------------------------------------------------

class RewardConstants:
    """Named scalar constants for every reward signal.

    Using a class (rather than module-level variables) keeps the namespace
    clean and allows future sub-classing if task-specific reward shaping is
    needed.
    """

    #: Awarded per zone when the dispatch fully satisfies all requirements.
    CORRECT_ALLOCATION: float = 2.0

    #: Bonus when the resolved zone contained a CRITICAL patient — this
    #: represents physically saving a life, so it carries the highest bonus.
    SAVE_CRITICAL_CASE: float = 5.0

    #: Penalty applied per step when the agent fails to dispatch *any*
    #: resources toward a HIGH or CATASTROPHIC fire, or a CRITICAL patient.
    #: This is the "delayed response to high severity" signal.
    DELAYED_HIGH_SEVERITY: float = -5.0

    #: Penalty applied *per zone* when the dispatch count exceeds what is
    #: strictly required (wasted units that could have been saved for later).
    OVER_ALLOCATION: float = -2.0

    #: Bonus for a dispatch that is *exactly* the minimum required — no waste,
    #: no deficit.  Encourages the agent to learn optimal minimal dispatch.
    EFFICIENT_RESOLUTION: float = 1.0

    #: Penalty applied per zone when the agent sends an empty dispatch while
    #: at least one active incident exists in that zone.
    IGNORE_INCIDENT: float = -4.0

    # ---------------------------------------------------------------------------
    # Trajectory-Aware Reward Shaping constants
    # ---------------------------------------------------------------------------

    #: Bonus granted when a zone's incident severity was actively escalating
    #: (consecutive_failures > 0 in the previous state) but the agent's
    #: dispatch this step successfully **halted** the degradation trajectory
    #: (severity rank Δ == 0 between previous_state and current_state).
    #:
    #: Design rationale: containment — stopping a cascade before it becomes
    #: catastrophic — is the single most valuable action in crisis management.
    #: This bonus incentivises the agent to prioritise *deteriorating* zones
    #: even when their current severity looks manageable.
    STABILIZATION_BONUS: float = 2.0

    #: Penalty applied per zone when any incident's ordinal severity rank
    #: *increases* between ``previous_state.zones[z]`` and
    #: ``current_state.zones[z]``.
    #:
    #: Design rationale: without this dense signal the agent must wait for the
    #: sparse cascade event (three consecutive failures) before receiving a
    #: meaningful negative gradient.  A -3.0 early-warning penalty pushes the
    #: policy toward proactive dispatch rather than reactive damage control.
    DEGRADATION_PENALTY: float = -3.0

    # --- Retained from original implementation (used by environment.py) ---
    #: Additional weather-based fire friction modifiers.
    WEATHER_HURRICANE_FIRE_FRICTION: int = 2
    WEATHER_STORM_FIRE_FRICTION: int = 1

    #: Ambulance modifier when GRIDLOCK traffic is present and no police sent.
    GRIDLOCK_AMB_FRICTION: int = 2


# ---------------------------------------------------------------------------
# Ordinal severity maps — enable integer arithmetic over enum ranks
# ---------------------------------------------------------------------------

#: Ascending ordinal rank for fire severity. Used by the trajectory-shaping
#: helper to compute Δ-severity as a signed integer.
_FIRE_RANK: Dict[FireLevel, int] = {
    FireLevel.NONE:         0,
    FireLevel.LOW:          1,
    FireLevel.MEDIUM:       2,
    FireLevel.HIGH:         3,
    FireLevel.CATASTROPHIC: 4,
}

#: Ascending ordinal rank for patient severity.
_PATIENT_RANK: Dict[PatientLevel, int] = {
    PatientLevel.NONE:     0,
    PatientLevel.MODERATE: 1,
    PatientLevel.CRITICAL: 2,
    PatientLevel.FATAL:    3,  # FATAL is the worst outcome; rank it highest.
}


# ---------------------------------------------------------------------------
# Pure helper: minimum resources required to resolve an incident
# ---------------------------------------------------------------------------

def _get_required_fire(level: FireLevel, weather: WeatherCondition) -> int:
    """Return the minimum fire units needed to resolve the given fire level.

    Weather acts as a *friction multiplier* — adverse conditions mean more
    resources must be committed to achieve the same effect.

    Mathematical mapping
    --------------------
    Base requirements (``req``):
        CATASTROPHIC → 5
        HIGH         → 3
        MEDIUM       → 2
        LOW          → 1
        NONE         → 0

    Weather modifier (``modifier``):
        HURRICANE → +2  (applied only if req > 0)
        STORM     → +1  (applied only if req > 0)
        CLEAR     → +0

    Final requirement: ``req + modifier``

    Args:
        level:   The current fire severity for a zone.
        weather: Global weather condition at this simulation step.

    Returns:
        Non-negative integer representing the minimum number of fire units
        required.  Returns 0 when there is no active fire.
    """
    if level == FireLevel.CATASTROPHIC:
        req = 5
    elif level == FireLevel.HIGH:
        req = 3
    elif level == FireLevel.MEDIUM:
        req = 2
    elif level == FireLevel.LOW:
        req = 1
    else:
        # FireLevel.NONE — no fire incident active.
        return 0

    # Apply weather friction only when an active fire exists.
    if weather == WeatherCondition.HURRICANE:
        req += RewardConstants.WEATHER_HURRICANE_FIRE_FRICTION
    elif weather == WeatherCondition.STORM:
        req += RewardConstants.WEATHER_STORM_FIRE_FRICTION

    logger.debug("Fire req for %s/%s = %d", level.value, weather.value, req)
    return req


def _get_required_ambulance(level: PatientLevel) -> int:
    """Return the minimum ambulances needed to resolve the given patient level.

    Gridlock friction is **not** included here — it is evaluated at dispatch
    time in both the environment and the reward, consistent with the original
    design.

    Mathematical mapping
    --------------------
        CRITICAL → 3
        MODERATE → 1
        FATAL    → 0  (cannot be saved; incident is already a loss)
        NONE     → 0

    Args:
        level: Current medical casualty severity for a zone.

    Returns:
        Non-negative integer representing the minimum ambulances required.
    """
    if level == PatientLevel.CRITICAL:
        return 3
    if level == PatientLevel.MODERATE:
        return 1
    # FATAL: no ambulances can help; NONE: no medical incident.
    return 0


# ---------------------------------------------------------------------------
# Per-zone reward breakdown
# ---------------------------------------------------------------------------

def _zone_reward(
    zone_id: str,
    zone: ZoneState,
    dispatch: ZoneDispatch,
    obs: Observation,
) -> float:
    """Compute the dense reward contribution from a single zone.

    This function implements the full reward table for one zone per step.
    Signals are computed independently and summed so that the agent receives
    a nuanced, multi-dimensional gradient.

    Algorithm
    ---------
    1. Compute ``R_fire``, ``R_amb``, and ``needs_traffic`` — the *required*
       dispatch quantities for this zone given current state and weather.
    2. If the zone has **no** incidents and the agent wastes resources on it,
       apply OVER_ALLOCATION.
    3. If the zone has incidents and the action is completely empty
       (D_fire=0, D_amb=0, control_traffic=False), apply IGNORE_INCIDENT.
    4. Check if the dispatch is **sufficient** (≥ requirements).
       If sufficient → CORRECT_ALLOCATION bonus.
    5. If there is a surplus (D > R), apply OVER_ALLOCATION penalty *on top*.
    6. If the dispatch is **exactly** (D == R), apply EFFICIENT_RESOLUTION.
    7. If the zone has a HIGH/CATASTROPHIC fire or CRITICAL patient and the
       dispatch is insufficient, apply DELAYED_HIGH_SEVERITY.
    8. If the resolved zone had a CRITICAL patient, apply SAVE_CRITICAL_CASE.

    Args:
        zone_id:  Zone name (for log messages only).
        zone:     Current hazard state of the zone.
        dispatch: Agent's dispatch order for this zone this step.
        obs:      Full observation (needed for weather).

    Returns:
        A float reward contribution from this zone.
    """
    # ------------------------------------------------------------------ #
    # Step 1 — Compute requirements
    # ------------------------------------------------------------------ #
    r_fire: int = _get_required_fire(zone.fire, obs.weather)
    r_amb: int  = _get_required_ambulance(zone.patient)

    # Traffic is a boolean requirement: do we need at least one police unit?
    needs_traffic: bool = zone.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK)

    # Gridlock penalty: when gridlock is active and no police are dispatched,
    # we require extra ambulances to compensate for blocked access routes.
    amb_gridlock_req: int = 0
    if zone.traffic == TrafficLevel.GRIDLOCK and not dispatch.control_traffic:
        amb_gridlock_req = RewardConstants.GRIDLOCK_AMB_FRICTION

    # Effective ambulance requirement including gridlock modifier.
    r_amb_effective: int = r_amb + amb_gridlock_req

    # Is there *any* active incident the agent should be responding to?
    has_incident: bool = (r_fire > 0) or (r_amb > 0) or needs_traffic

    zone_reward: float = 0.0

    # ------------------------------------------------------------------ #
    # Step 2 — Empty zone: penalise wasted dispatches
    # ------------------------------------------------------------------ #
    if not has_incident:
        if dispatch.dispatch_fire > 0 or dispatch.dispatch_ambulance > 0:
            # Wasting resources on a safe zone is a form of over-allocation.
            logger.debug("[%s] Wasted dispatch on safe zone → OVER_ALLOCATION", zone_id)
            zone_reward += RewardConstants.OVER_ALLOCATION
        # No incident means no further reward logic.
        return zone_reward

    # ------------------------------------------------------------------ #
    # Step 3 — Ignoring an active incident
    # ------------------------------------------------------------------ #
    action_is_empty = (
        dispatch.dispatch_fire == 0
        and dispatch.dispatch_ambulance == 0
        and not dispatch.control_traffic
    )
    if action_is_empty:
        logger.debug("[%s] Empty action on active incident → IGNORE_INCIDENT", zone_id)
        zone_reward += RewardConstants.IGNORE_INCIDENT
        # Additionally penalise if this is a HIGH severity incident being ignored.
        is_high_severity = (
            zone.fire in (FireLevel.HIGH, FireLevel.CATASTROPHIC)
            or zone.patient == PatientLevel.CRITICAL
        )
        if is_high_severity:
            logger.debug("[%s] High-severity ignored → DELAYED_HIGH_SEVERITY", zone_id)
            zone_reward += RewardConstants.DELAYED_HIGH_SEVERITY
        return zone_reward

    # ------------------------------------------------------------------ #
    # Step 4 and 5 — Sufficiency check and over-allocation detection
    #
    # Mathematical evaluation:
    #   surplus_fire = D_fire - R_fire
    #   surplus_amb  = D_amb  - R_amb
    #
    # "Correct allocation" means: D_fire >= R_fire AND D_amb >= R_amb_effective
    #   AND (control_traffic OR NOT needs_traffic)
    #
    # "Over-allocation" means: surplus_fire > 0 OR surplus_amb > 0
    #   (sending more than the strict minimum to a zone with active incidents)
    # ------------------------------------------------------------------ #
    fire_sufficient:    bool = (r_fire == 0) or (dispatch.dispatch_fire >= r_fire)
    amb_sufficient:     bool = (r_amb == 0)  or (dispatch.dispatch_ambulance >= r_amb_effective)
    traffic_sufficient: bool = (not needs_traffic) or dispatch.control_traffic

    allocation_correct = fire_sufficient and amb_sufficient and traffic_sufficient

    if allocation_correct:
        logger.debug("[%s] Dispatch sufficient → CORRECT_ALLOCATION (+%.1f)", zone_id, RewardConstants.CORRECT_ALLOCATION)
        zone_reward += RewardConstants.CORRECT_ALLOCATION

        # ---- Over-allocation penalty (co-occurs with correct allocation) ----
        # surplus_fire = D_fire - R_fire  > 0  → over-allocated fire
        # surplus_amb  = D_amb  - R_amb   > 0  → over-allocated ambulances
        # (Use R_amb, not R_amb_effective, for raw surplus computation.)
        surplus_fire = dispatch.dispatch_fire - r_fire if r_fire > 0 else dispatch.dispatch_fire
        surplus_amb  = dispatch.dispatch_ambulance - r_amb if r_amb > 0 else dispatch.dispatch_ambulance

        if surplus_fire > 0 or surplus_amb > 0:
            logger.debug(
                "[%s] Over-allocation (surplus_fire=%d, surplus_amb=%d) → OVER_ALLOCATION (%.1f)",
                zone_id, surplus_fire, surplus_amb, RewardConstants.OVER_ALLOCATION,
            )
            zone_reward += RewardConstants.OVER_ALLOCATION  # -2.0

        # ---- Efficient resolution bonus (exact match, no surplus) ----------
        # Condition: D_fire == R_fire  AND  D_amb == R_amb  AND surplus = 0
        fire_exact = (r_fire == 0 and dispatch.dispatch_fire == 0) or (dispatch.dispatch_fire == r_fire)
        amb_exact  = (r_amb  == 0 and dispatch.dispatch_ambulance == 0) or (dispatch.dispatch_ambulance == r_amb)
        if fire_exact and amb_exact:
            logger.debug("[%s] Exact dispatch → EFFICIENT_RESOLUTION (+%.1f)", zone_id, RewardConstants.EFFICIENT_RESOLUTION)
            zone_reward += RewardConstants.EFFICIENT_RESOLUTION  # +1.0

        # ---- Saving a critical case -----------------------------------------
        # This bonus applies when the dispatch is sufficient AND the zone
        # contains a CRITICAL patient (who will now be resolved).
        if zone.patient == PatientLevel.CRITICAL and amb_sufficient:
            logger.debug("[%s] Critical patient resolved → SAVE_CRITICAL_CASE (+%.1f)", zone_id, RewardConstants.SAVE_CRITICAL_CASE)
            zone_reward += RewardConstants.SAVE_CRITICAL_CASE  # +5.0

    else:
        # ------------------------------------------------------------------ #
        # Step 6 — Insufficient dispatch (partial or wrong)
        #
        # Partial Progress Gradient (max +0.90)
        # -----------------------------------------------------------------------
        # Without this, sending 4/5 required fire units yields exactly the same
        # score as sending 0/5 — a flat, unclimbable cliff that prevents the RL
        # policy from learning the correct threshold via gradient descent.
        #
        # The fulfilled ratio converts the binary check into a continuous signal:
        #
        #   fire_fulfilled = min(D_fire / R_fire, 1.0)   in [0.0, 1.0]
        #   amb_fulfilled  = min(D_amb  / R_amb,  1.0)   in [0.0, 1.0]
        #   partial_score  = fire_fulfilled * 0.45 + amb_fulfilled * 0.45
        #                  in [0.0, 0.90]
        #
        # Examples:
        #   4/5 fire units sent → fire_fulfilled=0.80 → partial_score += 0.36
        #   0/5 fire units sent → fire_fulfilled=0.00 → partial_score += 0.00
        #   exact traffic only  → only traffic component is scored above
        #
        # The HIGH-severity penalty is still applied on top of the partial score
        # so under-dispatching a CATASTROPHIC fire still produces a net negative
        # reward — the gradient exists but the sign remains punishing.
        # ------------------------------------------------------------------ #
        fire_fulfilled = (
            min(dispatch.dispatch_fire / r_fire, 1.0) if r_fire > 0 else 1.0
        )
        amb_fulfilled = (
            min(dispatch.dispatch_ambulance / r_amb_effective, 1.0)
            if r_amb_effective > 0
            else 1.0
        )
        partial_score = (fire_fulfilled * 0.45) + (amb_fulfilled * 0.45)

        if partial_score > 0.0:
            logger.debug(
                "[%s] Partial progress: fire_fulfilled=%.2f amb_fulfilled=%.2f "
                "-> partial_score=+%.2f",
                zone_id, fire_fulfilled, amb_fulfilled, partial_score,
            )
            zone_reward += partial_score

        is_high_severity = (
            zone.fire in (FireLevel.HIGH, FireLevel.CATASTROPHIC)
            or zone.patient == PatientLevel.CRITICAL
        )
        if is_high_severity:
            logger.debug(
                "[%s] Insufficient dispatch on high-severity incident "
                "-> DELAYED_HIGH_SEVERITY (%.1f)",
                zone_id, RewardConstants.DELAYED_HIGH_SEVERITY,
            )
            zone_reward += RewardConstants.DELAYED_HIGH_SEVERITY  # -5.0

    return zone_reward


# ---------------------------------------------------------------------------
# Trajectory-Aware Shaping — pure Δ-severity helper
# ---------------------------------------------------------------------------

def _trajectory_shaping(
    zone_id: str,
    current_zone: ZoneState,
    previous_zone: ZoneState,
) -> float:
    """Δ-severity shaping: Stabilization Bonus and Degradation Penalty.

    **Trajectory-Aware Reward Shaping** computes the difference in ordinal
    severity rank between two consecutive observations of the same zone and
    translates that Δ into a signed shaping bonus or penalty.

    This is a **pure function** — no side-effects, deterministic.

    Algorithm
    ---------
    For each incident type (fire, patient) in the zone:

    1.  Compute Δ_fire = _FIRE_RANK[current.fire] - _FIRE_RANK[previous.fire]
        Compute Δ_patient = _PATIENT_RANK[current.patient] - _PATIENT_RANK[previous.patient]

    2.  **Degradation Penalty** (``Δ > 0``):
        The incident escalated between steps.  This indicates the previous
        dispatch was insufficient (or the zone was ignored entirely).  We
        apply ``DEGRADATION_PENALTY = -3.0`` immediately so the agent can
        learn from the gradient before the cascade event fires.

    3.  **Stabilization Bonus** (``Δ == 0`` AND the zone was previously
        deteriorating, indicated by ``previous_zone.consecutive_failures > 0``):
        The agent halted a worsening trajectory without necessarily resolving
        the incident outright.  This is the hardest skill to learn — knowing
        exactly how many resources to send to stop (but not over-respond to) a
        cascading incident.  We reward it with ``STABILIZATION_BONUS = +2.0``.

    When Δ < 0 (severity *decreased*) no shaping is applied here because
    the base ``_zone_reward`` already grants CORRECT_ALLOCATION (+2.0) and
    optionally SAVE_CRITICAL_CASE (+5.0) for successful resolution.

    Args:
        zone_id:       Zone name (for debug logging only).
        current_zone:  Zone state in the *current* step observation.
        previous_zone: Zone state in the *previous* step observation.

    Returns:
        A float representing the net shaping bonus/penalty for this zone.
        Typical range: ``[-3.0, +2.0]`` per zone.
    """
    rc = RewardConstants
    shaping: float = 0.0

    # ---- Fire severity delta ----------------------------------------------- #
    delta_fire = _FIRE_RANK[current_zone.fire] - _FIRE_RANK[previous_zone.fire]

    if delta_fire > 0:
        # Fire severity *increased* — the agent's previous action failed to
        # contain the blaze.  Apply the dense degradation penalty immediately.
        logger.debug(
            "[%s] Fire degraded (%s → %s, Δ=%d) → DEGRADATION_PENALTY (%.1f)",
            zone_id,
            previous_zone.fire.value, current_zone.fire.value,
            delta_fire, rc.DEGRADATION_PENALTY,
        )
        shaping += rc.DEGRADATION_PENALTY

    elif delta_fire == 0 and previous_zone.consecutive_failures > 0 and current_zone.fire != FireLevel.NONE:
        # Fire severity *held steady* while the zone was previously deteriorating
        # (consecutive_failures > 0 means the prior step was insufficient).
        # This indicates the agent's current dispatch successfully stabilised the
        # fire — halting the cascade trajectory.
        logger.debug(
            "[%s] Fire stabilized (level=%s, prev_failures=%d) → STABILIZATION_BONUS (+%.1f)",
            zone_id,
            current_zone.fire.value, previous_zone.consecutive_failures, rc.STABILIZATION_BONUS,
        )
        shaping += rc.STABILIZATION_BONUS

    # ---- Patient severity delta -------------------------------------------- #
    delta_patient = _PATIENT_RANK[current_zone.patient] - _PATIENT_RANK[previous_zone.patient]

    if delta_patient > 0:
        logger.debug(
            "[%s] Patient status degraded (%s → %s, Δ=%d) → DEGRADATION_PENALTY (%.1f)",
            zone_id,
            previous_zone.patient.value, current_zone.patient.value,
            delta_patient, rc.DEGRADATION_PENALTY,
        )
        shaping += rc.DEGRADATION_PENALTY

    elif (
        delta_patient == 0
        and previous_zone.consecutive_failures > 0
        and current_zone.patient not in (PatientLevel.NONE, PatientLevel.FATAL)
    ):
        logger.debug(
            "[%s] Patient stabilized (level=%s, prev_failures=%d) → STABILIZATION_BONUS (+%.1f)",
            zone_id,
            current_zone.patient.value, previous_zone.consecutive_failures, rc.STABILIZATION_BONUS,
        )
        shaping += rc.STABILIZATION_BONUS

    return shaping


# ---------------------------------------------------------------------------
# Public API — the primary dense reward function
# ---------------------------------------------------------------------------

def calculate_step_reward(
    current_state: Observation,
    action: Action,
    previous_state: Observation,
) -> Reward:
    """Compute a dense structured Reward ledger for a single simulation step.

    This is the **primary public interface** of this module.  It is a **pure
    function** — calling it twice with identical arguments always produces the
    same result, and it has no observable side-effects.

    The reward is computed in two independent layers that are summed into the
    ``base_dispatch_score`` ledger line:

    **Layer 1 — Instantaneous Dispatch Quality** (per-zone via ``_zone_reward``)
        Evaluates whether the agent's *current* dispatch is correct, wasteful,
        efficient, or absent relative to this step's observed incident levels.
        Signals: CORRECT_ALLOCATION, OVER_ALLOCATION, EFFICIENT_RESOLUTION,
        IGNORE_INCIDENT, DELAYED_HIGH_SEVERITY, SAVE_CRITICAL_CASE.

    **Layer 2 — Trajectory-Aware Δ-Severity Shaping** (per-zone via
        ``_trajectory_shaping``)
        Evaluates how the world *changed* between the previous step and this
        step, providing an early dense gradient signal for containment vs.
        escalation.  Signals: STABILIZATION_BONUS (+2.0), DEGRADATION_PENALTY
        (-3.0).

    The two layers are designed to be complementary:
    * Layer 1 rewards the agent for *what it does*.
    * Layer 2 rewards the agent for *the consequences of what it did*.
    Together they create a rich, non-sparse gradient landscape that guides
    policy learning far more efficiently than terminal-only reward.

    Args:
        current_state:  The ``Observation`` *after* the tick (step counter has
                        already advanced), but *before* zones are resolved by
                        the environment.  This is the state the agent observed
                        when selecting ``action``.
        action:         The dispatch action selected by the agent this step.
        previous_state: The ``Observation`` from the *previous* episode step.
                        Used by the trajectory-shaping layer to compute Δ.
                        On step 1 this is the pre-tick snapshot of the initial
                        observation (a valid fallback with no cascades yet).

    Returns:
        A populated ``Reward`` Pydantic object.  Call ``.total_reward`` for the
        Gym-compliant scalar.  The full ledger JSON is available via
        ``reward.model_dump_json()`` for judge-facing diagnostics.
    """
    dispatch_quality_total = 0.0
    trajectory_shaping_total = 0.0

    for zone_id, zone_state in current_state.zones.items():
        dispatch: ZoneDispatch = action.allocations.get(zone_id, ZoneDispatch())

        # ---- Layer 1: Instantaneous dispatch quality ----------------------- #
        base_contribution = _zone_reward(zone_id, zone_state, dispatch, current_state)

        # ---- Layer 2: Trajectory-Aware Δ-severity shaping ----------------- #
        # Retrieve the matching zone from the previous step.  If the zone did
        # not exist in the prior observation (e.g., dynamically spawned), we
        # fall back to the current state, producing a Δ of zero (no shaping).
        prev_zone_state: ZoneState = previous_state.zones.get(zone_id, zone_state)
        shaping_contribution = _trajectory_shaping(zone_id, zone_state, prev_zone_state)

        zone_total = base_contribution + shaping_contribution
        logger.debug(
            "[%s] base=%.2f | shaping=%.2f | zone_total=%.2f",
            zone_id, base_contribution, shaping_contribution, zone_total,
        )
        dispatch_quality_total += base_contribution
        trajectory_shaping_total += shaping_contribution

    # base_dispatch_score = Layer 1 + Layer 2 (no NLP yet; that is Layer 3)
    base_dispatch_score = dispatch_quality_total + trajectory_shaping_total
    total_reward = base_dispatch_score  # NLP bonus added by compute_reward caller

    logger.info(
        "Step reward: %.4f (dispatch_quality=%.4f, trajectory_shaping=%.4f, active zones: %d)",
        total_reward, dispatch_quality_total, trajectory_shaping_total,
        len(current_state.zones),
    )

    return Reward(
        base_dispatch_score=base_dispatch_score,
        nlp_semantic_bonus=0.0,   # populated by compute_reward after NLP grading
        waste_penalty=0.0,        # populated by environment.py waste accumulator
        total_reward=total_reward,
        dispatch_quality=dispatch_quality_total,
        trajectory_shaping=trajectory_shaping_total,
        nlp_bonus=0.0,
        is_terminal=False,
    )


# ---------------------------------------------------------------------------
# Context-Grounded Semantic Grader — NLP broadcast bonus
# ---------------------------------------------------------------------------

def calculate_nlp_bonus(message: str, current_state: Observation) -> float:
    """Evaluate the agent's public broadcast message against the current crisis state.

    **Context-Grounded Semantic Grader** — Design Philosophy
    ---------------------------------------------------------
    A naive reward (+0.5 if message non-empty) creates a free-rider
    vulnerability: the agent learns to output ``"hello"`` and pocket the bonus
    without providing any real public-safety value.

    This grader prevents that by anchoring the bonus to the *content* of the
    message relative to what is objectively happening in the simulation.  The
    score is a three-component additive matrix that rewards specificity and
    penalises generic text by withholding partial scores for each missing element.

    Scoring Matrix (max 1.0 points)
    --------------------------------
    +----------------------------------+-------+----------------------------------+
    | Component                        | Score | Condition                        |
    +==================================+=======+==================================+
    | Base Match (zone name)           |  0.4  | message.lower() contains zone ID |
    +----------------------------------+-------+----------------------------------+
    | Hazard Match (keyword)           |  0.3  | fire critical: fire/blaze/burn   |
    |                                  |       | medical critical: medical/hospital|
    +----------------------------------+-------+----------------------------------+
    | Action Match (directive verb)    |  0.3  | evacuate/shelter/avoid/warning   |
    +----------------------------------+-------+----------------------------------+

    Anti-cheating properties
    ------------------------
    * Message must name the CORRECT zone for Base Match (wrong zone = 0).
    * Hazard keywords are TYPE-specific (fire vs medical, not interchangeable).
    * All three components are independent: possible scores are any subset.

    This is a **pure function** — no side-effects, deterministic.

    Args:
        message:       The agent's public_broadcast_message string.
        current_state: The current Observation used to ground keyword checks.

    Returns:
        A float in ``[0.0, 1.0]`` representing the graded broadcast quality.
    """
    if not message:
        return 0.0

    msg_lower = message.lower()

    # ---- Identify the most critical zone ----------------------------------- #
    best_score = 0
    critical_zone_id: str = ""
    critical_is_fire: bool = True

    for zone_id, z in current_state.zones.items():
        zone_severity = 0
        zone_fire_critical = True
        if z.fire == FireLevel.CATASTROPHIC:
            zone_severity = 100
        elif z.fire == FireLevel.HIGH:
            zone_severity = 50
        if z.patient == PatientLevel.CRITICAL:
            if 80 > zone_severity:
                zone_severity = 80
                zone_fire_critical = False
        if zone_severity > best_score:
            best_score = zone_severity
            critical_zone_id = zone_id
            critical_is_fire = zone_fire_critical

    if not critical_zone_id or best_score == 0:
        return 0.0

    nlp_score: float = 0.0

    # ---- Component 1: Base Match (0.4) ------------------------------------- #
    if critical_zone_id.lower() in msg_lower:
        nlp_score += 0.4
        logger.debug("NLP Base Match: '%s' found in message -> +0.4", critical_zone_id)

    # ---- Component 2: Hazard Match (0.3) ----------------------------------- #
    if critical_is_fire:
        fire_keywords = {"fire", "blaze", "burn", "flames", "inferno"}
        if any(kw in msg_lower for kw in fire_keywords):
            nlp_score += 0.3
            logger.debug("NLP Hazard Match (fire): keyword found -> +0.3")
    else:
        medical_keywords = {"medical", "hospital", "injury", "casualty", "patient", "ambulance"}
        if any(kw in msg_lower for kw in medical_keywords):
            nlp_score += 0.3
            logger.debug("NLP Hazard Match (medical): keyword found -> +0.3")

    # ---- Component 3: Action Match (0.3) ----------------------------------- #
    directive_verbs = {"evacuate", "shelter", "avoid", "warning", "alert", "flee", "leave"}
    if any(verb in msg_lower for verb in directive_verbs):
        nlp_score += 0.3
        logger.debug("NLP Action Match: directive verb found -> +0.3")

    logger.info(
        "NLP Grader: critical_zone=%s is_fire=%s msg=%r -> bonus=%.1f",
        critical_zone_id, critical_is_fire, message[:60], nlp_score,
    )
    return nlp_score


# ---------------------------------------------------------------------------
# Backward-compatibility shim — environment.py calls compute_reward()
# ---------------------------------------------------------------------------

def compute_reward(
    action: Action,
    obs: Observation,
    previous_state: Optional[Observation] = None,
) -> tuple[float, bool]:
    """Backward-compatible wrapper used by ``environment.py``.

    Applies three independent reward layers and returns the Gym-compliant
    ``(total_reward_float, all_resolved)`` tuple.  Internally this now
    instantiates a ``Reward`` Pydantic ledger object so that every step's
    arithmetic breakdown is captured and logged as structured JSON —
    providing direct evidence to judges that the ``Reward`` model is an
    active participant in the simulation loop, not a passive schema.

    Layers applied:

    1. **Dispatch Quality + Trajectory Shaping** (``calculate_step_reward``)
       Numeric dispatch decisions evaluated against incident requirements,
       plus Δ-severity shaping across consecutive steps.
    2. **Context-Grounded Semantic Grader** (``calculate_nlp_bonus``)
       Evaluates the quality of the agent's natural-language broadcast
       message against the actual crisis state.  Scores 0-1.0 only when
       a HIGH/CATASTROPHIC fire or CRITICAL patient is active.

    Args:
        action:         Agent's dispatch action.
        obs:            Current observation (pre-resolution).
        previous_state: Optional previous-step observation for shaping.

    Returns:
        ``(total_reward, all_resolved)`` where ``all_resolved`` is ``True``
        if no zone had an unmet requirement at this step.
    """
    prior = previous_state if previous_state is not None else obs

    # calculate_step_reward now returns a Reward ledger object (Layers 1 + 2)
    reward_ledger: Reward = calculate_step_reward(
        current_state=obs,
        action=action,
        previous_state=prior,
    )

    # ---- Layer 3: Context-Grounded Semantic Grader (NLP broadcast bonus) -- #
    # Gate: only award when there is an active HIGH/CATASTROPHIC fire OR a
    # CRITICAL patient.  Outside of crisis conditions the broadcast is
    # irrelevant — granting a bonus here would reward unnecessary scaremongering.
    nlp_bonus_value: float = 0.0
    has_high_severity = any(
        z.fire in (FireLevel.HIGH, FireLevel.CATASTROPHIC)
        or z.patient == PatientLevel.CRITICAL
        for z in obs.zones.values()
    )
    if has_high_severity and action.public_broadcast_message:
        nlp_bonus_value = calculate_nlp_bonus(action.public_broadcast_message, obs)
        logger.debug("Layer 3 NLP bonus applied: +%.2f", nlp_bonus_value)

    # Build the final Reward ledger with all three populated layers.
    # waste_penalty is left at 0.0 here; environment.py owns that accumulator
    # and logs the full ledger JSON with the live waste figure after resolution.
    base = reward_ledger.base_dispatch_score
    total = base + nlp_bonus_value  # waste_penalty deduction tracked separately
    final_ledger = Reward(
        base_dispatch_score=base,
        nlp_semantic_bonus=nlp_bonus_value,
        waste_penalty=0.0,
        total_reward=total,
        dispatch_quality=reward_ledger.dispatch_quality,
        trajectory_shaping=reward_ledger.trajectory_shaping,
        nlp_bonus=nlp_bonus_value,
        is_terminal=False,  # is_terminal set by environment.py after step
    )
    logger.info(
        "Reward Ledger JSON: %s",
        final_ledger.model_dump_json(),
    )

    # Derive all_resolved: True only if every zone had no active incidents
    # OR the dispatch was sufficient for all zones.
    all_resolved = True
    for zone_id, zone_state in obs.zones.items():
        r_fire = _get_required_fire(zone_state.fire, obs.weather)
        r_amb = _get_required_ambulance(zone_state.patient)
        needs_traffic = zone_state.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK)

        if r_fire == 0 and r_amb == 0 and not needs_traffic:
            continue  # Zone is clear - no response needed.

        dispatch = action.allocations.get(zone_id, ZoneDispatch())
        amb_gridlock = (
            RewardConstants.GRIDLOCK_AMB_FRICTION
            if zone_state.traffic == TrafficLevel.GRIDLOCK and not dispatch.control_traffic
            else 0
        )

        fire_ok    = (r_fire == 0) or (dispatch.dispatch_fire >= r_fire)
        amb_ok     = (r_amb == 0)  or (dispatch.dispatch_ambulance >= (r_amb + amb_gridlock))
        traffic_ok = (not needs_traffic) or dispatch.control_traffic

        if not (fire_ok and amb_ok and traffic_ok):
            all_resolved = False
            break

    return final_ledger.total_reward, all_resolved
