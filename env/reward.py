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
* **Backward-compatible shim** — ``compute_reward`` is retained as an alias
  so that ``environment.py``, which already imports it, continues to work
  without modification.

Reward Table (exact specification)
------------------------------------
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
from typing import Tuple, Dict

from env.models import (
    Action,
    Observation,
    FireLevel,
    PatientLevel,
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

    # --- Retained from original implementation (used by environment.py) ---
    #: Additional weather-based fire friction modifiers.
    WEATHER_HURRICANE_FIRE_FRICTION: int = 2
    WEATHER_STORM_FIRE_FRICTION: int = 1

    #: Ambulance modifier when GRIDLOCK traffic is present and no police sent.
    GRIDLOCK_AMB_FRICTION: int = 2


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
        # ------------------------------------------------------------------ #
        is_high_severity = (
            zone.fire in (FireLevel.HIGH, FireLevel.CATASTROPHIC)
            or zone.patient == PatientLevel.CRITICAL
        )
        if is_high_severity:
            logger.debug(
                "[%s] Insufficient dispatch on high-severity incident → DELAYED_HIGH_SEVERITY (%.1f)",
                zone_id, RewardConstants.DELAYED_HIGH_SEVERITY,
            )
            zone_reward += RewardConstants.DELAYED_HIGH_SEVERITY  # -5.0

    return zone_reward


# ---------------------------------------------------------------------------
# Public API — the primary dense reward function
# ---------------------------------------------------------------------------

def calculate_step_reward(
    current_state: Observation,
    action: Action,
    previous_state: Observation,  # noqa: ARG001 — reserved for future shaping
) -> float:
    """Compute a dense scalar reward for a single simulation step.

    This is the **primary public interface** of this module. It is a **pure
    function** — calling it twice with identical arguments always produces the
    same result, and it has no observable side-effects.

    The reward is computed by summing per-zone contributions evaluated
    independently via ``_zone_reward``.  See module docstring for the complete
    reward table.

    Args:
        current_state:  The ``Observation`` *after* the tick (step counter has
                        already advanced), but *before* zones are resolved by
                        the environment.  This is the state the agent observed
                        when selecting ``action``.
        action:         The dispatch action selected by the agent this step.
        previous_state: The ``Observation`` from the *previous* step.  Retained
                        for future temporal shaping (e.g., rewarding sustained
                        improvement), but not used in the current formula.

    Returns:
        A floating-point scalar reward. There are no explicit bounds, but
        practical per-step values lie roughly in ``[-9.0, +8.0]`` per zone.
    """
    total_reward = 0.0

    for zone_id, zone_state in current_state.zones.items():
        dispatch: ZoneDispatch = action.allocations.get(zone_id, ZoneDispatch())
        contribution = _zone_reward(zone_id, zone_state, dispatch, current_state)
        logger.debug("[%s] Zone reward contribution: %.2f", zone_id, contribution)
        total_reward += contribution

    logger.info("Step reward: %.4f (active zones: %d)", total_reward, len(current_state.zones))
    return total_reward


# ---------------------------------------------------------------------------
# Backward-compatibility shim — environment.py calls compute_reward()
# ---------------------------------------------------------------------------

def compute_reward(action: Action, obs: Observation) -> tuple[float, bool]:
    """Backward-compatible wrapper used by ``environment.py``.

    Delegates to ``calculate_step_reward`` and derives ``all_resolved`` by
    checking whether any zone still has active incidents after the reward
    computation.

    Args:
        action: Agent's dispatch action.
        obs:    Current observation (pre-resolution).

    Returns:
        ``(total_reward, all_resolved)`` where ``all_resolved`` is ``True``
        if no zone had an unmet requirement at this step.
    """
    total_reward = calculate_step_reward(
        current_state=obs,
        action=action,
        previous_state=obs,  # No previous state available via this shim.
    )

    # Derive all_resolved: True only if every zone had no active incidents
    # OR the dispatch was sufficient for all zones.
    all_resolved = True
    for zone_id, zone_state in obs.zones.items():
        r_fire = _get_required_fire(zone_state.fire, obs.weather)
        r_amb = _get_required_ambulance(zone_state.patient)
        needs_traffic = zone_state.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK)

        if r_fire == 0 and r_amb == 0 and not needs_traffic:
            continue  # Zone is clear — no response needed.

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

    return total_reward, all_resolved
