"""
env/grader.py
=============
Deterministic post-episode grader for the Adaptive Crisis Management Environment.

Design Contract
---------------
* **100% deterministic** — given the same episode metrics, ``grade_episode``
  always returns the identical float.  No randomness, no mutable global state.
* **Strictly bounded output** — the final score is always in ``[0.0, 1.0]``.
  ``math.inf``, ``math.nan``, and negative values are clamped automatically.
* **Three-component formula** (exact specification):

      score = (0.5 × success_rate) + (0.3 × efficiency) + (0.2 × resource_usage)

  Each sub-component is independently clamped to ``[0.0, 1.0]`` before the
  weighted sum, so the final score is guaranteed to be in ``[0.0, 1.0]``.

Sub-component Definitions
--------------------------
success_rate (weight 0.50):
    Fraction of all incidents that were resolved before the episode ended.

        success_rate = incidents_resolved / total_incidents

    Edge case: if ``total_incidents == 0``, success_rate = 1.0 (perfect score
    on an empty problem is the correct decision).

efficiency (weight 0.30):
    Normalised score capturing how quickly and precisely the agent responded.
    A "theoretical optimal" baseline is defined as resolving every incident in
    exactly ``OPTIMAL_STEPS_PER_INCIDENT`` steps with no wasted resources.

    Raw efficiency is derived from the cumulative step-reward relative to the
    best achievable reward for the given number of incidents:

        optimal_reward  = total_incidents × OPTIMAL_REWARD_PER_INCIDENT
        worst_reward    = total_incidents × WORST_REWARD_PER_INCIDENT
        reward_range    = optimal_reward - worst_reward  (> 0 by construction)

        raw_efficiency  = (total_reward - worst_reward) / reward_range

    Raw efficiency is then clamped to [0.0, 1.0].

    OPTIMAL_REWARD_PER_INCIDENT:
        Best achievable per-zone-per-step reward = +2.0 (correct alloc)
        + 1.0 (efficient) + 5.0 (save critical) = 8.0 theoretical max.
        We use a conservative 8.0.

    WORST_REWARD_PER_INCIDENT:
        Worst per-zone-per-step = –4.0 (ignore) + –5.0 (delayed high) = –9.0.
        We use –9.0.

resource_usage (weight 0.20):
    Measures how efficiently resources were used — penalising wasted dispatches
    and over-allocation throughout the episode.

        resource_usage = 1.0 - clamp(wasted_dispatches / max_possible_waste)

    ``wasted_dispatches`` is tracked by the environment as the running count
    of over-allocation events (steps where D > R for at least one resource).
    ``max_possible_waste`` is defined as:

        max_possible_waste = total_steps × len(zones)

    A value of 1.0 means perfect resource usage (zero waste); 0.0 means every
    step in every zone was an over-allocation.

    If ``wasted_dispatches`` is not tracked (legacy call), resource_usage
    defaults conservatively to 0.5.

Public API
----------
``grade_episode(...)``  — pure function, the primary interface.
``Grader``              — thin class wrapper that backends to ``grade_episode``.
                          Provides the ``get_score`` shim used by environment.py.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("crisis_env.grader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("[%(levelname)s] GRADER - %(message)s"))
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Grader configuration constants — single source of truth
# ---------------------------------------------------------------------------

class GraderConfig:
    """Named constants that parameterise the grading formula.

    All values are *mathematically* motivated; see module docstring for
    derivations.  Changing these will alter every grade. document the reason.
    """

    # Weighted formula coefficients (must sum to 1.0)
    WEIGHT_SUCCESS_RATE: float = 0.50
    WEIGHT_EFFICIENCY:   float = 0.30
    WEIGHT_RESOURCE_USE: float = 0.20

    # Efficiency normalisation anchors (see module docstring)
    OPTIMAL_REWARD_PER_INCIDENT: float  =  8.0  # best achievable per incident
    WORST_REWARD_PER_INCIDENT:   float  = -9.0  # worst possible per incident


# Sanity-check the weights — fail fast at import time if misconfigured.
assert abs(
    GraderConfig.WEIGHT_SUCCESS_RATE
    + GraderConfig.WEIGHT_EFFICIENCY
    + GraderConfig.WEIGHT_RESOURCE_USE
    - 1.0
) < 1e-9, "GraderConfig weights must sum to exactly 1.0"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``value`` into ``[lo, hi]``, handling NaN and Inf gracefully.

    Args:
        value: The value to clamp.
        lo:    Lower bound (inclusive).
        hi:    Upper bound (inclusive).

    Returns:
        A float in ``[lo, hi]``.
    """
    if math.isnan(value) or math.isinf(value):
        logger.warning("Non-finite value %.6g clamped to lo=%.2f", value, lo)
        return lo
    return max(lo, min(hi, value))


def _compute_success_rate(incidents_resolved: int, total_incidents: int) -> float:
    """Fraction of total incidents that were successfully resolved.

    Mathematical definition:
        success_rate = incidents_resolved / total_incidents    (if total > 0)
        success_rate = 1.0                                     (if total == 0)

    Args:
        incidents_resolved: Count of incidents cleared before episode end.
        total_incidents:    Count of incidents present at episode start.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    if total_incidents <= 0:
        # No incidents to resolve → agent succeeded trivially.
        logger.debug("total_incidents=0; success_rate defaults to 1.0")
        return 1.0

    raw = float(incidents_resolved) / float(total_incidents)
    result = _clamp(raw)
    logger.debug("success_rate = %d/%d = %.4f", incidents_resolved, total_incidents, result)
    return result


def _compute_efficiency(total_reward: float, total_incidents: int) -> float:
    """Normalised efficiency score derived from the episode's cumulative reward.

    The formula linearly maps ``total_reward`` from the range
    ``[worst, optimal]`` into ``[0.0, 1.0]`` using the anchors defined in
    ``GraderConfig``.

    Mathematical derivation:
        optimal_reward = total_incidents × OPTIMAL_REWARD_PER_INCIDENT
        worst_reward   = total_incidents × WORST_REWARD_PER_INCIDENT
        reward_range   = optimal_reward - worst_reward           (always > 0)

        raw_efficiency = (total_reward - worst_reward) / reward_range
        efficiency     = clamp(raw_efficiency, 0.0, 1.0)

    If ``total_incidents == 0`` there is no reward to earn, so we return 1.0
    (the agent did nothing wrong on an empty problem).

    Args:
        total_reward:    Cumulative floating-point reward from the episode.
        total_incidents: Number of incidents active at episode start.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    if total_incidents <= 0:
        logger.debug("total_incidents=0; efficiency defaults to 1.0")
        return 1.0

    cfg = GraderConfig
    optimal_reward = float(total_incidents) * cfg.OPTIMAL_REWARD_PER_INCIDENT
    worst_reward   = float(total_incidents) * cfg.WORST_REWARD_PER_INCIDENT
    reward_range   = optimal_reward - worst_reward  # Guaranteed > 0 by construction.

    if reward_range <= 0.0:
        # Defensive guard; should never trigger with current config values.
        logger.error(
            "reward_range=%.4g ≤ 0; efficiency defaults to 0.0 (config error).",
            reward_range,
        )
        return 0.0

    raw_efficiency = (total_reward - worst_reward) / reward_range
    result = _clamp(raw_efficiency)
    logger.debug(
        "efficiency: total_reward=%.2f, worst=%.2f, optimal=%.2f, range=%.2f → %.4f",
        total_reward, worst_reward, optimal_reward, reward_range, result,
    )
    return result


def _compute_resource_usage(
    wasted_dispatches: Optional[int],
    total_steps: int,
    num_zones: int,
) -> float:
    """Normalised resource efficiency score (1.0 = no waste, 0.0 = maximum waste).

    Mathematical definition:
        max_possible_waste = total_steps × num_zones
        resource_usage     = 1.0 - clamp(wasted / max_possible_waste)

    A "wasted dispatch" is any step-zone combination where the agent dispatched
    more units than were required (over-allocation), as tracked by the
    environment's ``OVER_ALLOCATION`` reward event.

    Args:
        wasted_dispatches: Number of over-allocation events across the episode,
                           or ``None`` if not tracked by the caller.
        total_steps:       Total steps in the episode.
        num_zones:         Number of distinct zones in the environment.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    if wasted_dispatches is None:
        # Legacy path: caller did not track waste. Use 0.5 as a neutral prior.
        logger.debug("wasted_dispatches not provided; resource_usage defaults to 0.5")
        return 0.5

    max_waste = float(max(total_steps * num_zones, 1))  # Prevent division by zero.
    waste_ratio = _clamp(float(wasted_dispatches) / max_waste)
    result = 1.0 - waste_ratio
    logger.debug(
        "resource_usage: wasted=%d, max_waste=%.0f, ratio=%.4f → score=%.4f",
        wasted_dispatches, max_waste, waste_ratio, result,
    )
    return result


# ---------------------------------------------------------------------------
# Primary public function — the deterministic grader
# ---------------------------------------------------------------------------

def grade_episode(
    incidents_resolved: int,
    total_incidents: int,
    total_reward: float,
    total_steps: int,
    num_zones: int,
    wasted_dispatches: Optional[int] = None,
) -> float:
    """Deterministically grade a complete episode and return a score in [0.0, 1.0].

    Implements the exact formula specified by the hackathon:

        score = (0.5 × success_rate) + (0.3 × efficiency) + (0.2 × resource_usage)

    Each sub-component is independently clamped to [0.0, 1.0] before the
    weighted sum, guaranteeing the final result is in [0.0, 1.0].

    This function is **pure** and **deterministic**:
        - No global mutable state.
        - No randomness.
        - Same inputs → same output, always.

    Args:
        incidents_resolved: Number of distinct incidents cleared this episode.
        total_incidents:    Number of distinct incidents present at episode
                            start (denominator for success_rate).
        total_reward:       Cumulative step-reward sum for the episode.
        total_steps:        Total number of steps taken (used for efficiency
                            and resource_usage normalisation).
        num_zones:          Number of zones in the environment (used to derive
                            the max possible waste ceiling).
        wasted_dispatches:  Optional count of over-allocation events tracked
                            by the environment.  If ``None``, resource_usage
                            defaults to 0.5.

    Returns:
        A float in ``[0.0, 1.0]`` representing the episode quality.

    Example:
        >>> grade_episode(
        ...     incidents_resolved=3,
        ...     total_incidents=4,
        ...     total_reward=18.5,
        ...     total_steps=10,
        ...     num_zones=3,
        ...     wasted_dispatches=2,
        ... )
        0.7...  # exact value depends on config
    """
    # ---- Sub-component 1: success_rate ----------------------------------- #
    success_rate = _compute_success_rate(incidents_resolved, total_incidents)

    # ---- Sub-component 2: efficiency ------------------------------------- #
    efficiency = _compute_efficiency(total_reward, total_incidents)

    # ---- Sub-component 3: resource_usage --------------------------------- #
    resource_usage = _compute_resource_usage(wasted_dispatches, total_steps, num_zones)

    # ---- Weighted sum (exact specification formula) ----------------------- #
    cfg = GraderConfig
    raw_score = (
        (cfg.WEIGHT_SUCCESS_RATE * success_rate)
        + (cfg.WEIGHT_EFFICIENCY   * efficiency)
        + (cfg.WEIGHT_RESOURCE_USE * resource_usage)
    )

    # Final clamp ensures the output is STRICTLY in [0.0, 1.0].
    final_score = _clamp(raw_score)

    logger.info(
        "Grade: success_rate=%.4f(×%.1f) + efficiency=%.4f(×%.1f) + "
        "resource_usage=%.4f(×%.1f) → raw=%.4f → score=%.4f",
        success_rate,  cfg.WEIGHT_SUCCESS_RATE,
        efficiency,    cfg.WEIGHT_EFFICIENCY,
        resource_usage, cfg.WEIGHT_RESOURCE_USE,
        raw_score, final_score,
    )
    return final_score


# ---------------------------------------------------------------------------
# Grader class — thin wrapper for backward compatibility with environment.py
# ---------------------------------------------------------------------------

class GraderException(Exception):
    """Raised when grading logic encounters an unrecoverable state."""


class Grader:
    """Thin class wrapper around the ``grade_episode`` pure function.

    ``environment.py`` calls ``Grader().get_score(resolved, total, reward)``
    at every step.  This class retains that API while delegating all real
    computation to the deterministic ``grade_episode`` function.

    Note:
        ``get_score`` is called mid-episode with partial metrics. The
        ``resource_usage`` sub-component therefore defaults to 0.5 (neutral
        prior) because the environment does not pass wasted-dispatch counts
        through this legacy interface.
    """

    def get_score(
        self,
        incidents_resolved: int,
        total_incidents: int,
        total_reward: float,
        total_steps: int = 1,
        num_zones: int = 3,
        wasted_dispatches: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute a partial episode score and return ``(score, efficiency)``.

        Delegates to ``grade_episode``.  The ``efficiency`` sub-component is
        returned as the second element for ``environment.py``'s ``info`` dict.

        Args:
            incidents_resolved: Resolved-so-far count.
            total_incidents:    Episode-start incident count.
            total_reward:       Cumulative reward so far.
            total_steps:        Steps elapsed (defaults to 1 for partial runs).
            num_zones:          Number of zones (defaults to 3).
            wasted_dispatches:  Optional over-allocation event count.

        Returns:
            ``(final_score, efficiency)`` — both floats in ``[0.0, 1.0]``.
        """
        try:
            efficiency = _compute_efficiency(total_reward, total_incidents)
            final_score = grade_episode(
                incidents_resolved=incidents_resolved,
                total_incidents=total_incidents,
                total_reward=total_reward,
                total_steps=total_steps,
                num_zones=num_zones,
                wasted_dispatches=wasted_dispatches,
            )
            return final_score, efficiency
        except Exception as exc:
            logger.error("Grader.get_score failed: %s", exc)
            raise GraderException(f"Grading failed: {exc}") from exc
