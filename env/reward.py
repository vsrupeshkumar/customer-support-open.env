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

    If Δ == 0  AND  previous_failures[zone_id] > 0  → the agent halted the
    degradation.  Grant +2.0.

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

from env.logger import get_engine_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = get_engine_logger("crisis_env.reward")


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
    #: (previous_failures > 0 in the previous state) but the agent's
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
        # Anti-Exploit Enforced: Unified Dimensional Penalty. Prevents 'Zero-Cost Action' free-riding
        # by ensuring all resource deployments (including boolean support actions like police)
        # incur strict waste penalties in safe zones.
        if dispatch.dispatch_fire > 0 or dispatch.dispatch_ambulance > 0 or dispatch.control_traffic:
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

def _trajectory_reward(
    zone_id: str,
    current_zone: ZoneState,
    previous_zone: ZoneState,
    previous_failures: Optional[Dict[str, int]] = None,
) -> float:
    """Evaluate temporally shaped reward components (Δ values).

    Detects if incidents worsened (DEGRADATION_PENALTY) or if a worsening
    incident was successfully halted (STABILIZATION_BONUS).

    **Stabilization Logic**:
    If Δ == 0 (severity held steady) and the zone was *previously*
    deteriorating, indicated by ``previous_failures[zone_id] > 0``:
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
        previous_failures: Dictionary mapping zone IDs to their failure
                       counters from the *previous* step.

    Returns:
        A float representing the net shaping bonus/penalty for this zone.
        Typical range: ``[-3.0, +2.0]`` per zone.
    """
    rc = RewardConstants
    shaping: float = 0.0
    
    prev_fails = 0
    if previous_failures is not None:
        prev_fails = previous_failures.get(zone_id, 0)

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

    elif delta_fire == 0 and prev_fails > 0 and current_zone.fire != FireLevel.NONE:
        # Fire severity *held steady* while the zone was previously deteriorating.
        # This indicates the agent's current dispatch successfully stabilised the
        # fire — halting the cascade trajectory.
        logger.debug(
            "[%s] Fire stabilized (level=%s, prev_failures=%d) → STABILIZATION_BONUS (+%.1f)",
            zone_id,
            current_zone.fire.value, prev_fails, rc.STABILIZATION_BONUS,
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
        and prev_fails > 0
        and current_zone.patient not in (PatientLevel.NONE, PatientLevel.FATAL)
    ):
        logger.debug(
            "[%s] Patient stabilized (level=%s, prev_failures=%d) → STABILIZATION_BONUS (+%.1f)",
            zone_id,
            current_zone.patient.value, prev_fails, rc.STABILIZATION_BONUS,
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
    previous_failures: Optional[Dict[str, int]] = None,
    step_count: int = 1,
) -> Reward:
    """Compute a dense structured Reward ledger for a single simulation step.

    This is the **primary public interface** of this module.  It is a **pure
    function** — calling it twice with identical arguments always produces the
    same result, and it has no observable side-effects.

    The reward is computed in three complementary layers:

    **Layer 1 — Instantaneous Dispatch Quality** (per-zone via ``_zone_reward``)
        Evaluates whether the agent's *current* dispatch is correct, wasteful,
        efficient, or absent relative to this step's observed incident levels.
        Signals: CORRECT_ALLOCATION, OVER_ALLOCATION, EFFICIENT_RESOLUTION,
        IGNORE_INCIDENT, DELAYED_HIGH_SEVERITY, SAVE_CRITICAL_CASE.

    **Layer 2 — Trajectory-Aware Δ-Severity Shaping** (per-zone via
        ``_trajectory_reward``)
        Evaluates how the world *changed* between the previous step and this
        step, providing an early dense gradient signal for containment vs.
        escalation.  Signals: STABILIZATION_BONUS (+2.0), DEGRADATION_PENALTY
        (-3.0).

    **Layer 3 — Multi-Objective POMDP Formula** (globally authoritative scalar)
        The canonical crisis-objective reward defined here — NOT in app.py.
        This enforces strict POMDP architectural purity: the API layer is a
        dumb router; all math lives in the environment.

            severity_delta    = Σ (previous_rank - current_rank) over all zones
                                (fire + patient dimensions)
            efficiency_bonus  = (resources_saved / total_resources) × 0.5
            time_penalty      = 0.1   (constant per-step cost)
            multi_obj_reward  = (severity_delta × 1.5) + efficiency_bonus
                                − time_penalty

        Final reward = Layer1 + Layer2 + Layer3 (unified, single authority).

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
        previous_failures: Internal environment dictionary mapping zone IDs
                           to their failure counters from the *previous* step.

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
        shaping_contribution = _trajectory_reward(
            zone_id, zone_state, prev_zone_state, previous_failures
        )

        zone_total = base_contribution + shaping_contribution
        logger.debug(
            "[%s] base=%.2f | shaping=%.2f | zone_total=%.2f",
            zone_id, base_contribution, shaping_contribution, zone_total,
        )
        dispatch_quality_total += base_contribution
        trajectory_shaping_total += shaping_contribution

    # base_dispatch_score = Layer 1 + Layer 2
    base_dispatch_score = dispatch_quality_total + trajectory_shaping_total

    # =========================================================================
    # Layer 3 — Multi-Objective POMDP Formula (canonical, authoritative)
    #
    # ALL reward math lives here, in the environment layer — NOT in app.py.
    # This enforces strict POMDP architectural purity (no split-brain).
    #
    # Severity ordinal rank map (shared across fire and patient dimensions):
    #   none        → 0
    #   low         → 1
    #   moderate    → 2  (patient)  |  medium → 2  (fire)
    #   high        → 3             |  critical → 4
    #   catastrophic→ 5             |  fatal   → 5
    # =========================================================================
    # Step A — Aggregate severity delta (previous_severity - current_severity)
    # A positive delta means the world improved (reward ↑).
    # A negative delta means it degraded (reward ↓).
    # Both fire and patient dimensions are summed for a holistic signal.
    # BUG-019: Coherent Severity Ordinal Rank substitution
    severity_delta: float = 0.0
    for zone_id, cur_z in current_state.zones.items():
        prev_z = previous_state.zones.get(zone_id, cur_z)
        severity_delta += _FIRE_RANK[prev_z.fire] - _FIRE_RANK[cur_z.fire]
        severity_delta += _PATIENT_RANK[prev_z.patient] - _PATIENT_RANK[cur_z.patient]

    # Step B — Resource efficiency bonus
    # resources_saved = units NOT deployed this step (proxy for conservation).
    # efficiency_bonus = (resources_saved / total_resources) * 0.5
    total_resources: float = float(
        previous_state.idle_resources.fire_units
        + previous_state.idle_resources.ambulances
        + previous_state.idle_resources.police
    )
    resources_used: float = float(
        sum(
            d.dispatch_fire + d.dispatch_ambulance + (1 if d.control_traffic else 0)
            for d in action.allocations.values()
        )
    )
    resources_saved: float = max(0.0, total_resources - resources_used)
    efficiency_bonus: float = (
        (resources_saved / total_resources) * 0.5
        if total_resources > 0 else 0.0
    )

    # Step C — Base time penalty (encourages episode efficiency)
    time_penalty: float = 0.1

    # Canonical Multi-Objective scalar:
    #   R_multi = (severity_delta × 1.5) + efficiency_bonus − time_penalty
    multi_obj_reward: float = (severity_delta * 1.5) + efficiency_bonus - time_penalty

    # 1. Synthesize the complete Multi-Objective Reward Tensor
    #    R_total = R_base + R_semantic - R_waste + R_efficiency - R_time + R_multiobj
    #    NOTE: nlp_semantic_bonus and waste_penalty are 0.0 at this stage;
    #          they are injected by compute_reward / environment.py respectively.
    #          They are included here explicitly so that the Pydantic ledger
    #          identity (verify_reward_ledger) holds at construction time.
    
    # Mathematical integration of the POMDP Temporal Discount Factor (γ = 0.99)
    # BUG-011 FIX: Temporarily removed from here to prevent double-discounting.
    # Discount applies for step t >= 1 exactly ONCE to the final total_reward scalar
    # inside environment.py's step() aggregation.
    # The pure ledger constants are left perfectly static.
    total_reward = (
        base_dispatch_score +
        0.0 -           # nlp_semantic_bonus  (orphan-safe; populated downstream)
        0.0 +           # waste_penalty        (orphan-safe; populated downstream)
        efficiency_bonus -
        time_penalty +
        multi_obj_reward
    )

    # 2. SANITIZATION LAYER: Round all floats to 4 decimal places for LLM token
    #    efficiency. Strips IEEE 754 artifacts (e.g. -17.200000000000003) from
    #    the JSON payload before Pydantic validation. The model_validator uses
    #    math.isclose(abs_tol=1e-4) and remains untouched — rounding here is
    #    strictly a presentation-layer concern upstream of the ledger.
    base_dispatch_score    = round(base_dispatch_score, 4)
    nlp_semantic_bonus_r   = round(0.0, 4)           # placeholder; populated downstream
    waste_penalty_r        = round(0.0, 4)            # placeholder; populated downstream
    efficiency_bonus       = round(efficiency_bonus, 4)
    time_penalty           = round(time_penalty, 4)
    multi_obj_reward       = round(multi_obj_reward, 4)
    # Recompute total from rounded sub-components to prevent IEEE 754 drift
    # from exceeding the Pydantic model_validator's abs_tol=1e-4 threshold.
    # With 5 zones the accumulated drift can exceed 1e-4 if total is rounded
    # independently from its constituents.
    total_reward = round(
        base_dispatch_score + nlp_semantic_bonus_r - waste_penalty_r
        + efficiency_bonus - time_penalty + multi_obj_reward, 4
    )

    logger.info(
        "Step reward total=%.4f | dispatch_quality=%.4f trajectory_shaping=%.4f "
        "severity_delta=%.2f efficiency_bonus=%.4f time_penalty=%.2f multi_obj=%.4f "
        "active_zones=%d",
        total_reward,
        dispatch_quality_total, trajectory_shaping_total,
        severity_delta, efficiency_bonus, time_penalty, multi_obj_reward,
        len(current_state.zones),
    )

    # 3. Construct the strict Pydantic ledger
    step_reward = Reward(
        base_dispatch_score=base_dispatch_score,
        nlp_semantic_bonus=nlp_semantic_bonus_r,  # populated by compute_reward after NLP grading
        waste_penalty=waste_penalty_r,             # populated by environment.py waste accumulator
        efficiency_bonus=efficiency_bonus,
        time_penalty=time_penalty,
        multi_obj=multi_obj_reward,
        total_reward=total_reward,
        dispatch_quality=dispatch_quality_total,
        trajectory_shaping=trajectory_shaping_total,
        nlp_bonus=0.0,
        is_terminal=False,
    )
    return step_reward


# Directive 3 Compliance: Penalties are mathematically subtracted from the total reward.
# Zero-floor clamps have been removed to prevent keyword stuffing and resource spamming.
def calculate_nlp_bonus(message: str, current_state: Observation) -> float:
    """Evaluate the agent's public broadcast message against the current crisis state.

    Directive 3 Compliance: Ruthless Utility NLP Grader — no zero-floor clamping.
    The bonus CAN be negative if the agent hallucinates keywords or spams words.

    **Precision-Recall Penalty Model** — Design Philosophy
    -------------------------------------------------------
    The legacy additive grader was vulnerable to keyword stuffing: an agent
    that outputs every possible hazard keyword guaranteed a perfect score
    regardless of relevance.  This function uses:

        Final_NLP_Bonus = (Keywords_Found × Weight)
                        - (Hallucinated_Keywords × λ)
                        - (Word_Count × γ)

    Where:
        Valid    = keywords that are ACTUALLY active in the current ZoneState.
        Invalid  = tracked keywords present in message but NOT active (false positives).
        w_i      = per-component weight (0.4 zone / 0.3 hazard / 0.3 action).
        λ        = 0.5  (hallucination penalty per false-positive keyword).
        γ        = 0.01 (bloat penalty per word beyond the 50-word threshold).

    Anti-Bloat Constraint (Word-Count Based)
    -----------------------------------------
    Word count beyond 50 words is penalised at γ = 0.01 per excess word.
    This creates a continuous negative gradient against verbosity / spamming,
    replacing the legacy binary character-length clamp.

    Zero-Floor Removal
    ------------------
    The zero-floor clamp ``max(0.0, score)`` has been REMOVED.  The NLP
    sub-reward IS allowed to go negative if the agent hallucinates or spams.
    This enforces mathematical precision — forfeiting the bonus is no longer
    a safe fall-back for a poorly-calibrated agent.

    Scoring Matrix (max 1.0 points, before penalties)
    --------------------------------------------------
    +---------------------------------+-------+-------------------------------------+
    | Component                       | Score | Condition                           |
    +=================================+=======+=====================================+
    | Base Match (zone name)          |  0.4  | Message names the most critical zone |
    +---------------------------------+-------+-------------------------------------+
    | Hazard Match (active type only) |  0.3  | Correct hazard keyword for zone type |
    +---------------------------------+-------+-------------------------------------+
    | Action Match (directive verb)   |  0.3  | Actionable directive present         |
    +---------------------------------+-------+-------------------------------------+

    Penalty Matrix
    --------------
    +-----------------------------+-------------+-----------------------------------------------+
    | Penalty                     | Amount      | Trigger                                       |
    +=============================+=============+===============================================+
    | Hallucination (per keyword) |  -λ = -0.5  | Tracked keyword in message but NOT active     |
    +-----------------------------+-------------+-----------------------------------------------+
    | Bloat (per excess word)     |  -γ = -0.01 | Each word beyond the 50-word threshold        |
    +-----------------------------+-------------+-----------------------------------------------+

    Args:
        message:       The agent's public_broadcast_message string.
        current_state: The current Observation used to ground keyword checks.

    Returns:
        A float representing the graded broadcast quality.  CAN BE NEGATIVE
        if hallucination or bloat penalties outweigh positive components.
        There is NO zero-floor clamp (Directive 3 compliance).
    """
    if not message:
        return 0.0

    msg_lower = message.lower()

    # =========================================================================
    # BUILD KEYWORD UNIVERSE: every tracked keyword across the entire system.
    # These are the words an agent can be penalised for hallucinating.
    # =========================================================================
    ALL_ZONE_KEYWORDS: frozenset[str] = frozenset(
        z_id.lower() for z_id in current_state.zones
    )
    ALL_FIRE_KEYWORDS: frozenset[str] = frozenset(
        {"fire", "blaze", "burn", "flames", "inferno"}
    )
    ALL_MEDICAL_KEYWORDS: frozenset[str] = frozenset(
        {"medical", "hospital", "injury", "casualty", "patient", "ambulance"}
    )
    ALL_TRAFFIC_KEYWORDS: frozenset[str] = frozenset(
        {"gridlock", "traffic", "congestion", "blockage"}
    )
    ALL_TRACKED: frozenset[str] = (
        ALL_ZONE_KEYWORDS | ALL_FIRE_KEYWORDS | ALL_MEDICAL_KEYWORDS | ALL_TRAFFIC_KEYWORDS
    )

    # =========================================================================
    # BUILD ACTIVE KEYWORD SET: keywords warranted by the CURRENT ZoneState.
    # Only these keywords earn positive scores; all others are hallucinations.
    # =========================================================================
    active_keywords: set[str] = set()

    # Identify the single most critical zone (highest severity).
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
        elif z.fire == FireLevel.MEDIUM:
            zone_severity = 25

        if z.patient == PatientLevel.CRITICAL and 80 > zone_severity:
            zone_severity = 80
            zone_fire_critical = False

        # Populate active keyword set for this zone.
        if z.fire not in (FireLevel.NONE,):
            active_keywords.update(ALL_FIRE_KEYWORDS)
        if z.patient not in (PatientLevel.NONE, PatientLevel.FATAL):
            active_keywords.update(ALL_MEDICAL_KEYWORDS)
        if z.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK):
            active_keywords.update(ALL_TRAFFIC_KEYWORDS)
        
        if zone_severity > 0:
            active_keywords.add(zone_id.lower())

        if zone_severity > best_score:
            best_score = zone_severity
            critical_zone_id = zone_id
            critical_is_fire = zone_fire_critical

    if not critical_zone_id or best_score == 0:
        # No active crisis → no valid bonus available.
        return 0.0

    # =========================================================================
    # ANTI-BLOAT CONSTRAINT (Directive 3): word-count based, γ = 0.01 per
    # excess word beyond the 50-word threshold.  Creates a continuous negative
    # gradient against verbosity — replacing the legacy binary char-length clamp.
    # =========================================================================
    _GAMMA: float = 0.01          # bloat penalty per excess word
    _WORD_LIMIT: int = 50         # words allowed before bloat penalty kicks in
    word_count: int = len(message.split())
    excess_words: int = max(0, word_count - _WORD_LIMIT)
    bloat_penalty: float = -_GAMMA * excess_words
    if excess_words > 0:
        logger.warning(
            "NLP Anti-Bloat (Directive 3): %d words, %d excess (limit=%d) → bloat_penalty=%.3f",
            word_count, excess_words, _WORD_LIMIT, bloat_penalty,
        )

    # =========================================================================
    # PRECISION-RECALL SCORING: positive weights for valid hits.
    # =========================================================================
    nlp_score: float = 0.0

    import re
    # ---- Component 1: Base Match (0.4) — must name the correct critical zone — #
    if re.search(rf"\b{re.escape(critical_zone_id.lower())}\b", msg_lower):
        nlp_score += 0.4
        logger.debug("NLP Base Match: '%s' found → +0.4", critical_zone_id)

    # ---- Component 2: Hazard Match (0.3) — correct hazard type only --- #
    if critical_is_fire:
        if any(kw in msg_lower for kw in ALL_FIRE_KEYWORDS):
            nlp_score += 0.3
            logger.debug("NLP Hazard Match (fire) → +0.3")
    else:
        if any(kw in msg_lower for kw in ALL_MEDICAL_KEYWORDS):
            nlp_score += 0.3
            logger.debug("NLP Hazard Match (medical) → +0.3")

    # ---- Component 3: Action Match (0.3) — directive verbs only -------- #
    _DIRECTIVE_VERBS = frozenset({"evacuate", "shelter", "avoid", "warning", "alert", "flee", "leave"})
    if any(verb in msg_lower for verb in _DIRECTIVE_VERBS):
        nlp_score += 0.3
        logger.debug("NLP Action Match: directive verb found → +0.3")

    # =========================================================================
    # FALSE POSITIVE SCANNER (λ = 0.5 per hallucinated keyword).
    # Directive 3: hallucination_penalty is subtracted with NO zero-floor.
    # =========================================================================
    _LAMBDA: float = 0.5
    hallucination_count: int = 0

    for tracked_kw in ALL_TRACKED:
        if tracked_kw in msg_lower and tracked_kw not in active_keywords:
            hallucination_count += 1
            logger.debug(
                "NLP False Positive: keyword '%s' in message but NOT active → -%.1f",
                tracked_kw, _LAMBDA,
            )

    hallucination_penalty: float = _LAMBDA * hallucination_count

    # =========================================================================
    # FINAL SCORE (Directive 3): NO zero-floor clamp.
    #   Final_NLP_Bonus = (Keywords_Found × Weight)
    #                   - (Hallucinated_Keywords × λ)
    #                   - (Excess_Words × γ)
    # The score IS allowed to be negative — forfeiting the bonus is NOT a
    # safe strategy for an agent that hallucinates or keyword-stuffs.
    # =========================================================================
    final_score: float = nlp_score - hallucination_penalty + bloat_penalty
    # REMOVED: final_score = max(0.0, raw_score)  ← zero-floor clamp eliminated per Directive 3.

    logger.info(
        "NLP Grader (Directive 3) | critical_zone=%s is_fire=%s | "
        "keywords=%.1f hallucinations=%d(×λ=%.1f) words=%d(excess=%d,×γ=%.2f) | "
        "final_nlp_bonus=%.4f | msg=%r",
        critical_zone_id, critical_is_fire,
        nlp_score, hallucination_count, _LAMBDA,
        word_count, excess_words, _GAMMA,
        final_score,
        message[:60],
    )
    return final_score


# ---------------------------------------------------------------------------
