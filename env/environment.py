"""
env/environment.py
==================
Core OpenEnv-compliant RL environment for the Adaptive Crisis Management
simulation.

This module is the single authoritative implementation of the environment
interface.  It imports all data contracts from ``env.models`` and delegates
reward computation to ``env.reward``.  No UI, web, or dashboard logic is
permitted here.

Interface
---------
The public API follows the OpenEnv specification:

    env = CrisisManagementEnv(task_id=1)
    obs: Observation = env.reset(seed=42)
    obs, reward, done, info = env.step(action)
    state: EnvironmentState = env.state()
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # Used exclusively for np.random.seed() in reset()

from env.models import (
    Action,
    ActiveDeployment,
    EnvironmentState,
    FireLevel,
    Observation,
    PatientLevel,
    ResourcePool,
    TrafficLevel,
    WeatherCondition,
    ZoneDispatch,
    ZoneState,
)
from env.reward import compute_reward, calculate_step_reward
from env.tasks import Task, create_task

# ---------------------------------------------------------------------------
# Module-level logger (engine diagnostics only — no UI formatting)
# ---------------------------------------------------------------------------

logger = logging.getLogger("crisis_env.engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("[%(levelname)s] ENGINE - %(message)s"))
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EnvironmentException(Exception):
    """Raised when the environment encounters an unrecoverable state error.

    Examples:
        Calling ``step()`` after the episode has already terminated.
        Attempting to load an invalid task ID.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _LifecycleManager:
    """Manages the cooldown countdown for all active (deployed) unit batches.

    This is a stateless helper; all mutations happen to the mutable lists /
    ``Observation`` objects passed into its methods.
    """

    @staticmethod
    def tick(
        obs: Observation,
        active_deployments: List[ActiveDeployment],
    ) -> List[ActiveDeployment]:
        """Advance every deployment by one step and return still-active ones.

        Units whose ``steps_remaining`` reaches zero are returned to the
        ``idle_resources`` pool.

        Args:
            obs: The current observation whose resource pools will be mutated
                in-place to reflect returning units.
            active_deployments: The list of currently tracked deployments.

        Returns:
            A new list containing only deployments that still have steps
            remaining (i.e., units that have *not* yet returned).
        """
        still_active: List[ActiveDeployment] = []
        rec_fire = rec_amb = rec_pol = 0

        for dep in active_deployments:
            dep.steps_remaining -= 1
            if dep.steps_remaining <= 0:
                rec_fire += dep.fire_units
                rec_amb += dep.ambulances
                rec_pol += dep.police
            else:
                still_active.append(dep)

        if rec_fire or rec_amb or rec_pol:
            obs.idle_resources.fire_units += rec_fire
            obs.idle_resources.ambulances += rec_amb
            obs.idle_resources.police += rec_pol
            obs.busy_resources.fire_units -= rec_fire
            obs.busy_resources.ambulances -= rec_amb
            obs.busy_resources.police -= rec_pol
            logger.debug(
                "Recovered %dF | %dA | %dP to idle pool.", rec_fire, rec_amb, rec_pol
            )

        return still_active


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------

class CrisisManagementEnv:
    """OpenEnv-compliant multi-zone crisis management RL environment.

    The environment models a simulated city divided into named zones, each
    of which can simultaneously suffer from fire, medical, and traffic
    incidents.  An agent dispatches emergency resources each step and
    receives a shaped scalar reward.

    Attributes:
        task_id: Selected task difficulty (1 = Easy, 2 = Medium, 3 = Hard).

    Example:
        >>> env = CrisisManagementEnv(task_id=2)
        >>> obs = env.reset(seed=0)
        >>> action = Action(allocations={"Downtown": ZoneDispatch(dispatch_fire=3)})
        >>> obs, reward, done, info = env.step(action)
    """

    def __init__(self, task_id: int = 1, seed: Optional[int] = None) -> None:
        """Initialise the environment and load the specified task.

        Args:
            task_id: Difficulty level to load (1, 2, or 3).
            seed: Optional random seed for reproducibility.  If provided it
                is forwarded to ``reset()``.

        Raises:
            EnvironmentException: If ``task_id`` does not correspond to a
                valid registered task.
        """
        self.task_id: int = task_id

        try:
            self._task: Task = create_task(task_id)
        except ValueError as exc:
            raise EnvironmentException(
                f"Task ID {task_id} is not registered."
            ) from exc

        # Internal bookkeeping — initialised properly by reset()
        self._rng: random.Random = random.Random(seed)
        self._active_deployments: List[ActiveDeployment] = []
        self._total_incidents: int = 0
        self._resolved_incidents: int = 0
        self._lives_saved: int = 0
        self._total_reward: float = 0.0
        self._is_done: bool = False
        self.obs: Observation           # set by reset()
        self._prev_obs: Optional[Observation] = None  # Blocker #3: temporal shaping

        self.reset(seed=seed)
        logger.info("CrisisManagementEnv successfully booted locally against Task %d.", task_id)

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment to a fresh episode state.

        Determinism contract
        --------------------
        When ``seed`` is not ``None`` this method seeds **three** independent
        RNG layers so that every source of randomness in the episode is locked:

        1. ``random.seed(seed)``      — Python stdlib global (used by third-
                                        party libs that may call random.*)
        2. ``np.random.seed(seed)``   — NumPy global RNG (same rationale)
        3. ``random.Random(seed)``    — The environment's *private* instance
                                        (``self._rng``), used for any internal
                                        random decisions outside tasks.

        The task's ``generate_initial_observation(seed=seed)`` also receives
        the same seed and constructs its own *isolated* ``random.Random(seed)``
        so that incident generation is reproducible and immune to the global
        state set in steps 1–2 above.

        Args:
            seed: Optional integer seed for deterministic incident generation.
                Overrides the seed supplied at construction time.

        Returns:
            The initial ``Observation`` of the new episode.
        """
        if seed is not None:
            # ---- Absolute determinism: lock ALL RNG layers ---------------
            # 1. Python stdlib global — any code calling random.random() or
            #    random.choice() directly will produce the same sequence.
            random.seed(seed)

            # 2. NumPy global RNG — covers numpy operations in training loops,
            #    feature engineering pipelines, or observation wrappers.
            np.random.seed(seed)

            # 3. Private environment RNG instance — isolated, re-seeded so
            #    it does not share state with the global random module.
            self._rng = random.Random(seed)

            logger.debug(
                "Determinism enforced: random.seed(%d), np.random.seed(%d), "
                "self._rng = random.Random(%d).",
                seed, seed, seed,
            )

        self.obs = self._task.generate_initial_observation(seed=seed)
        self._active_deployments = []
        self._total_reward = 0.0
        self._is_done = False
        self._resolved_incidents = 0
        self._lives_saved = 0
        self._total_incidents = self._count_incidents(self.obs)
        self._wasted_dispatches: int = 0  # Blocker #2: tracks over-allocation events for grader.
        self._prev_obs: Optional[Observation] = None  # Blocker #3: temporal shaping anchor

        logger.debug("Environment reset.  Total incidents: %d.", self._total_incidents)
        return self.obs.model_copy(deep=True)

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute a single simulation step.

        Args:
            action: The agent's dispatching decisions for this step.

        Returns:
            A 4-tuple of:

            - **obs** (``Observation``): New observation after the step.
            - **reward** (``float``): Shaped reward for this step.
            - **done** (``bool``): Whether the episode has terminated.
            - **info** (``dict``): Auxiliary diagnostics including
              ``"score"``, ``"efficiency"``, ``"resolved"``, and ``"total"``.

        Raises:
            EnvironmentException: If called after the episode is already done.
        """
        if self._is_done:
            raise EnvironmentException(
                "Episode is done.  Call reset() before stepping again."
            )

        # 1. Advance time and recover returned units.
        # Blocker #3: snapshot previous obs BEFORE the tick so temporal
        # shaping in calculate_step_reward receives a genuinely distinct prior.
        prev_obs_snapshot: Observation = self.obs.model_copy(deep=True)

        self.obs.step += 1
        self._active_deployments = _LifecycleManager.tick(
            self.obs, self._active_deployments
        )

        # 2. Compute reward *before* resolving zones (uses pre-action state).
        # Blocker #3: pass distinct previous_state to unlock temporal shaping.
        reward = calculate_step_reward(
            current_state=self.obs,
            action=action,
            previous_state=prev_obs_snapshot,
        )
        self._total_reward += reward

        # 3. Commit allocations and resolve each zone.
        # Track over-allocations for Blocker #2 grader accuracy.
        for zone_id, zone_state in self.obs.zones.items():
            dispatch = action.allocations.get(zone_id, ZoneDispatch())
            used_fire, used_amb, used_pol = self._commit_allocation(
                zone_id, zone_state, dispatch
            )
            self._resolve_zone(zone_id, zone_state, used_fire, used_amb, used_pol)

            # Blocker #2: count over-allocation events (D > R for any resource).
            from env.reward import _get_required_fire, _get_required_ambulance
            req_f = _get_required_fire(zone_state.fire, self.obs.weather)
            req_a = _get_required_ambulance(zone_state.patient)
            if dispatch.dispatch_fire > req_f or dispatch.dispatch_ambulance > req_a:
                self._wasted_dispatches += 1

        # Advance temporal shaping anchor.
        self._prev_obs = prev_obs_snapshot

        # 4. Determine episode termination.
        all_clear = all(
            z.fire == FireLevel.NONE
            and z.patient in (PatientLevel.NONE, PatientLevel.FATAL)
            and z.traffic == TrafficLevel.LOW
            for z in self.obs.zones.values()
        )
        if all_clear or self.obs.step >= self.obs.max_steps:
            self._is_done = True
            logger.info("Evaluation Terminated natively. Executing Scorecard hook.")

        # 5. Build info dict — Blocker #2: all three grader components now live.
        from env.grader import Grader  # local import avoids circular deps at module level

        score, eff_score = Grader().get_score(
            incidents_resolved=self._resolved_incidents,
            total_incidents=self._total_incidents,
            total_reward=self._total_reward,
            total_steps=max(self.obs.step, 1),
            num_zones=len(self.obs.zones),
            wasted_dispatches=self._wasted_dispatches,
        )
        info: Dict[str, Any] = {
            "resolved": self._resolved_incidents,
            "total": self._total_incidents,
            "score": score,
            "efficiency": eff_score,
            "wasted_dispatches": self._wasted_dispatches,
        }

        return self.obs.model_copy(deep=True), float(reward), self._is_done, info

    def state(self) -> EnvironmentState:
        """Return a complete internal snapshot of the environment.

        This method exposes *more* than the public ``Observation``.  It is
        intended for graders, monitors, and testing harnesses — not for the
        agent itself during a live episode.

        Returns:
            A fully populated ``EnvironmentState`` reflecting the current
            internal variables.
        """
        from env.grader import Grader  # local import avoids circular deps at module level

        # Blocker #2: pass all three grader components for accurate resource_usage.
        score, eff_score = Grader().get_score(
            incidents_resolved=self._resolved_incidents,
            total_incidents=self._total_incidents,
            total_reward=self._total_reward,
            total_steps=max(self.obs.step, 1),
            num_zones=len(self.obs.zones),
            wasted_dispatches=self._wasted_dispatches,
        )
        return EnvironmentState(
            step_count=self.obs.step,
            max_steps=self.obs.max_steps,
            observation=self.obs.model_copy(deep=True),
            total_reward=self._total_reward,
            is_done=self._is_done,
            success=(self._resolved_incidents == self._total_incidents),
            metrics={
                "efficiency": eff_score,
                "lives_saved": float(self._lives_saved),
                "wasted_dispatches": float(self._wasted_dispatches),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        """Whether the current episode has terminated (read-only)."""
        return self._is_done

    @property
    def total_reward(self) -> float:
        """Cumulative episode reward accumulated so far (read-only)."""
        return self._total_reward

    @staticmethod
    def _count_incidents(obs: Observation) -> int:
        """Count total distinct active incidents across all zones.

        Args:
            obs: The initial observation from which to count.

        Returns:
            Integer count of active fire, medical, and traffic incidents.
        """
        count = 0
        for z in obs.zones.values():
            if z.fire != FireLevel.NONE:
                count += 1
            if z.patient != PatientLevel.NONE:
                count += 1
            if z.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK):
                count += 1
        return count

    def _commit_allocation(
        self,
        zone_id: str,
        zone_state: ZoneState,
        dispatch: ZoneDispatch,
    ) -> Tuple[int, int, int]:
        """Deduct dispatched units from idle pool and track as a deployment.

        Dispatch counts are clamped to the available idle pool; the agent
        cannot dispatch more resources than are currently idle.

        Args:
            zone_id: Identifier of the target zone (used for logging).
            zone_state: Current state of the zone (used for cooldown calc).
            dispatch: The dispatch order to commit.

        Returns:
            A 3-tuple ``(used_fire, used_amb, used_pol)`` reflecting the
            actual units dispatched after clamping.
        """
        used_fire = min(dispatch.dispatch_fire, self.obs.idle_resources.fire_units)
        used_amb = min(dispatch.dispatch_ambulance, self.obs.idle_resources.ambulances)
        used_pol = (
            1 if dispatch.control_traffic and self.obs.idle_resources.police > 0 else 0
        )

        # Move units from idle → busy pool.
        self.obs.idle_resources.fire_units -= used_fire
        self.obs.idle_resources.ambulances -= used_amb
        self.obs.idle_resources.police -= used_pol
        self.obs.busy_resources.fire_units += used_fire
        self.obs.busy_resources.ambulances += used_amb
        self.obs.busy_resources.police += used_pol

        # Calculate cooldown duration (weather + gridlock prolong deployments).
        cooldown = 1
        if self.obs.weather == WeatherCondition.HURRICANE:
            cooldown = 3
        elif self.obs.weather == WeatherCondition.STORM:
            cooldown = 2
        if zone_state.traffic == TrafficLevel.GRIDLOCK:
            cooldown += 2

        if used_fire or used_amb or used_pol:
            self._active_deployments.append(
                ActiveDeployment(
                    zone_id=zone_id,
                    fire_units=used_fire,
                    ambulances=used_amb,
                    police=used_pol,
                    steps_remaining=cooldown,
                )
            )
            logger.debug(
                "%s: committed %dF/%dA/%dP (cooldown=%d).",
                zone_id,
                used_fire,
                used_amb,
                used_pol,
                cooldown,
            )

        return used_fire, used_amb, used_pol

    def _resolve_zone(
        self,
        zone_id: str,
        zone_state: ZoneState,
        used_fire: int,
        used_amb: int,
        used_pol: int,
    ) -> None:
        """Evaluate whether dispatched units are sufficient to resolve the zone.

        If the dispatch satisfies all incident requirements the incidents are
        cleared.  Otherwise the zone's ``consecutive_failures`` counter is
        incremented and a cascade may be triggered.

        Args:
            zone_id: Zone identifier (used for cascade logging).
            zone_state: Mutable zone state that will be updated in-place.
            used_fire: Fire units actually committed this step.
            used_amb: Ambulance units actually committed this step.
            used_pol: Police units actually committed this step.
        """
        from env.reward import _get_required_fire, _get_required_ambulance  # pure fns

        req_fire = _get_required_fire(zone_state.fire, self.obs.weather)
        req_amb = _get_required_ambulance(zone_state.patient)
        req_traffic = zone_state.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK)

        has_active = req_fire > 0 or req_amb > 0 or req_traffic

        # Determine sufficiency.
        is_sufficient = True
        if req_fire > 0 and used_fire < req_fire:
            is_sufficient = False
        if req_amb > 0:
            gridlock_mod = (
                2
                if zone_state.traffic == TrafficLevel.GRIDLOCK and not used_pol
                else 0
            )
            if used_amb < req_amb + gridlock_mod:
                is_sufficient = False
        if req_traffic and not used_pol:
            is_sufficient = False

        # ---- Successful resolution ----------------------------------------
        if is_sufficient and has_active:
            if used_fire > 0 and zone_state.fire != FireLevel.NONE:
                zone_state.fire = FireLevel.NONE
                self._resolved_incidents += 1
            if used_amb > 0 and zone_state.patient not in (
                PatientLevel.NONE, PatientLevel.FATAL
            ):
                zone_state.patient = PatientLevel.NONE
                self._resolved_incidents += 1
                self._lives_saved += 18
            if used_pol > 0 and req_traffic:
                zone_state.traffic = TrafficLevel.LOW
                self._resolved_incidents += 1
            zone_state.consecutive_failures = 0

        # ---- Failure / cascading escalation ---------------------------------
        elif not is_sufficient and has_active:
            zone_state.consecutive_failures += 1
            if zone_state.consecutive_failures >= 3:
                logger.warning(
                    "CASCADING HAZARD TRIGGERED AT %s DUE TO LATENCY.", zone_id
                )
                self._escalate_zone(zone_state)
                zone_state.consecutive_failures = 0
        else:
            # No incidents — reset failure counter regardless.
            zone_state.consecutive_failures = 0

    @staticmethod
    def _escalate_zone(zone_state: ZoneState) -> None:
        """Advance all hazards in a zone by one severity level (cascade).

        Args:
            zone_state: The zone to escalate (mutated in-place).
        """
        _fire_escalation: Dict[FireLevel, FireLevel] = {
            FireLevel.LOW: FireLevel.MEDIUM,
            FireLevel.MEDIUM: FireLevel.HIGH,
            FireLevel.HIGH: FireLevel.CATASTROPHIC,
        }
        _patient_escalation: Dict[PatientLevel, PatientLevel] = {
            PatientLevel.MODERATE: PatientLevel.CRITICAL,
            PatientLevel.CRITICAL: PatientLevel.FATAL,
        }

        if zone_state.fire in _fire_escalation:
            zone_state.fire = _fire_escalation[zone_state.fire]
        if zone_state.patient in _patient_escalation:
            zone_state.patient = _patient_escalation[zone_state.patient]
        if zone_state.traffic == TrafficLevel.HEAVY:
            zone_state.traffic = TrafficLevel.GRIDLOCK
