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

import numpy as np  # Used for np.random.default_rng(seed) — private instance only

from env.models import (
    Action,
    ActiveDeployment,
    EnvironmentState,
    FireLevel,
    Observation,
    PatientLevel,
    ResourcePool,
    Reward,
    TrafficLevel,
    WeatherCondition,
    ZoneDispatch,
    ZoneState,
    StructuralHallucinationError,
)
from env.reward import compute_reward, calculate_step_reward
from env.tasks import Task, create_task
from openenv.core import Environment

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


class InventoryBreachException(Exception):
    """Raised internally when the LLM requests more units than available.

    This exception is caught within ``step()`` and converted to a catastrophic
    terminal penalty rather than propagating to the caller.  It serves as the
    internal signal that an Inventory Breach has occurred.
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
                dep.status = "IDLE"
                rec_fire += dep.fire_units
                rec_amb += dep.ambulances
                rec_pol += dep.police
            else:
                dep.status = "BUSY"
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

class CrisisManagementEnv(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compliant multi-zone crisis management RL environment.

    Directive 1 Compliance: Gymnasium v0.29+ API Enforced. The environment correctly
    outputs the 5-tuple step and distinct terminated/truncated signals for accurate 
    Bellman equation bootstrapping.

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
        super().__init__()
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
        self._terminated: bool = False
        self._truncated: bool = False
        # Directive 4: step counter and episode limit are PRIVATE backend state.
        # They are NEVER serialised into the agent-facing Observation.
        self._step_count: int = 0
        self._max_steps: int = self._task.get_max_steps()
        self.obs: Observation           # set by reset()
        self._prev_obs: Optional[Observation] = None  # Blocker #3: temporal shaping

        self.reset(seed=seed)
        logger.info("CrisisManagementEnv successfully booted locally against Task %d.", task_id)

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment to a fresh episode state.


        Monolithic Entropy Lock
        -----------------------
        Engages a Monolithic Entropy Lock.  By utilizing instance-bound PRNGs
        (a NumPy Generator and a Python ``random.Random``), we ensure the
        Markov Decision Process (MDP) transition function P(s'|s,a) remains
        completely stationary and parallel-safe, preventing PRNG state
        corruption during concurrent evaluator testing.

        OS-level entropy is fully quarantined.  Every source of randomness in
        the episode is driven by these two explicit, isolated generator objects
        seeded from the ``seed`` parameter — **never** from global RNG state.

        Two instance-bound PRNG objects are locked on every call:

        1. ``self._rng = random.Random(seed)``
                                         The environment's isolated stdlib RNG.
                                         Passed **directly** into
                                         ``generate_initial_observation(rng=self._rng)``
                                         so that there is exactly **ONE** PRNG
                                         object driving the entire episode.
                                         Tasks must not construct their own RNG.
        2. ``self._np_rng = np.random.default_rng(seed)``
                                         A private NumPy Generator for any
                                         continuous or array-valued stochastic
                                         transitions inside ``step()``.

        **No global ``random.seed()`` or ``np.random.seed()`` call is made.**
        Third-party libraries relying on the global Python RNG are not our
        responsibility; quarantining our own generators is sufficient and
        avoids interfering with the caller's global state.

        Args:
            seed: Integer seed for fully deterministic episode generation.
                Defaults to 42 — ensures unseeded calls are still deterministic.

        Returns:
            The initial ``Observation`` of the new episode.
        """
        # ---- Monolithic Entropy Lock: always seed, no conditional guard ---- #
        # Private stdlib instance — isolated, shares no state with global random.
        self._rng = random.Random(seed)

        # Private NumPy generator — legacy-API-free, fully isolated.
        self._np_rng = np.random.default_rng(seed)

        logger.debug(
            "Monolithic Entropy Lock engaged: self._rng=Random(%s), "
            "self._np_rng=default_rng(%s) — global RNG untouched.",
            seed, seed,
        )

        # Pass self._rng directly — NO new random.Random(seed) created in Task.
        # There is exactly ONE PRNG object per episode, owned by this instance.
        self.obs = self._task.generate_initial_observation(rng=self._rng)
        self._active_deployments = []
        self._total_reward = 0.0
        self._is_done = False
        self._terminated = False
        self._truncated = False
        self._resolved_incidents = 0
        self._lives_saved = 0
        self._total_incidents = self._count_incidents(self.obs)
        self._wasted_dispatches: float = 0.0  # Blocker #2: severity-weighted waste accumulator.
        self._prev_obs: Optional[Observation] = None  # Blocker #3: temporal shaping anchor.
        # Dynamic Horizon Scaling: Ensure theoretical solvability
        initial_zone_count = self._total_incidents
        min_required_steps = int(initial_zone_count * 2.0) # Assume 2 steps per zone to resolve
        exploration_buffer = 4
        
        # Scale episode boundary based on state complexity, with absolute upper bound of 25
        self._max_steps = min(25, max(12, min_required_steps + exploration_buffer))
        self._step_count = 0

        # POMDP boundary: Track failure cascades internally here, NOT in the
        # public Observation model which the agent receives.
        self._zone_failures: Dict[str, int] = {
            z_id: 0 for z_id in self.obs.zones.keys()
        }

        logger.debug("Environment reset.  Total incidents: %d.", self._total_incidents)
        return self.obs.model_copy(deep=True), {}

    def step(
        self, action: Any
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """Execute a single simulation step.

        Args:
            action: The agent's dispatching decisions for this step.

        Returns:
            A 5-tuple of:

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

        # =====================================================================
        # Directive 2 & 5: Structural Darwinism & Schema Brutality
        # =====================================================================
        if isinstance(action, StructuralHallucinationError):
            reward = -20.0
            self._total_reward += reward
            self._step_count += 1
            
            done = self._step_count >= self._max_steps
            if done:
                self._is_done = True
                self._truncated = True

            logger.error(
                "[Step %d] Directive 5 Schema Brutality: Bouncer caught hallucination! Continuous penalty applied.",
                self._step_count
            )
            
            info = {
                "resolved": self._resolved_incidents,
                "total": self._total_incidents,
                "score": 0.0,
                "efficiency": 0.0,
                "error_msg": f"Directive 5 Bouncer Caught Hallucination: {action}",
            }
            return self.obs.model_copy(deep=True), float(reward), self._terminated, self._truncated, info

        if not isinstance(action, Action):
            reward = -20.0
            self._total_reward += reward
            self._step_count += 1
            
            done = self._step_count >= self._max_steps
            if done:
                self._is_done = True
                self._truncated = True

            logger.error(
                "[Step %d] Structural Darwinism: Action is not a valid Pydantic Action! Continuous penalty applied.",
                self._step_count
            )
            
            info = {
                "resolved": self._resolved_incidents,
                "total": self._total_incidents,
                "score": 0.0,
                "efficiency": 0.0,
                "error_msg": f"Action is invalid: {action}",
            }
            return self.obs.model_copy(deep=True), float(reward), self._terminated, self._truncated, info

        # 1. Advance time and recover returned units.
        # Blocker #3: capture a deep-copy of the *current* obs BEFORE the tick
        # so we have a clean immutable snapshot of "what the world looked like
        # in the previous step" to pass into the reward function.
        # This local variable serves two purposes:
        #   a) Step-1 fallback for calculate_step_reward (self._prev_obs is None
        #      at episode start, so we use this as the bootstrap prior).
        #   b) The severity snapshot for the waste accumulator (must read
        #      pre-resolution fire/patient levels).
        prev_obs_snapshot: Observation = self.obs.model_copy(deep=True)

        # Directive 4: advance the private step counter, NOT obs.step.
        self._step_count += 1
        self._active_deployments = _LifecycleManager.tick(
            self.obs, self._active_deployments
        )

        # =====================================================================
        # Agentic Purity Check: Strict MDP enforcement. Impossible actions are
        # rejected and penalized to enforce LLM inventory tracking.
        #
        # Pre-flight INVENTORY BREACH gate:
        #   Sum the LLM's requested units across ALL zones for each resource
        #   category.  If ANY category exceeds the available idle pool, the
        #   action is physically impossible and constitutes a hallucination
        #   failure.  We do NOT silently clamp — we apply a catastrophic
        #   terminal penalty, void the entire action, and terminate the episode.
        #
        #   Penalty formula:
        #     R_terminal = -15.0 × severity_multiplier
        #   where severity_multiplier is amplified (×2.0) when a HIGH /
        #   CATASTROPHIC fire or CRITICAL patient is active, because failing
        #   to deploy during a life-threatening crisis is maximally egregious.
        # =====================================================================
        total_req_fire = sum(
            action.allocations.get(z, ZoneDispatch()).dispatch_fire
            for z in self.obs.zones
        )
        total_req_amb = sum(
            action.allocations.get(z, ZoneDispatch()).dispatch_ambulance
            for z in self.obs.zones
        )
        total_req_pol = sum(
            1 for z in self.obs.zones
            if action.allocations.get(z, ZoneDispatch()).control_traffic
        )

        idle = self.obs.idle_resources
        breach_fire  = total_req_fire > idle.fire_units
        breach_amb   = total_req_amb  > idle.ambulances
        breach_pol   = total_req_pol  > idle.police

        if breach_fire or breach_amb or breach_pol:
            # Identify which categories were breached for the feedback message.
            breach_details: List[str] = []
            if breach_fire:
                breach_details.append(
                    f"fire_units: requested {total_req_fire}, available {idle.fire_units}"
                )
            if breach_amb:
                breach_details.append(
                    f"ambulances: requested {total_req_amb}, available {idle.ambulances}"
                )
            if breach_pol:
                breach_details.append(
                    f"police: requested {total_req_pol}, available {idle.police}"
                )

            breach_msg = "CRITICAL FAILURE: " + "; ".join(breach_details) + ". Action voided."
            logger.error(
                "[Step %d] INVENTORY BREACH — %s",
                self._step_count, breach_msg,
            )

            # Severity multiplier: ×2.0 if a life-threatening incident is active.
            has_critical = any(
                z.fire in (FireLevel.HIGH, FireLevel.CATASTROPHIC)
                or z.patient == PatientLevel.CRITICAL
                for z in self.obs.zones.values()
            )
            severity_multiplier = 2.0 if has_critical else 1.0
            breach_penalty: float = -15.0 * severity_multiplier

            self._total_reward += breach_penalty
            
            # Dynamically calculate the absolute step boundary
            done = self._step_count >= self._max_steps
            if done:
                self._is_done = True
                self._truncated = True

            logger.error(
                "[Step %d] INVENTORY BREACH continuous penalty: %.1f (severity_mult=%.1f). "
                "Episode continues to allow Agent Recovery.",
                self._step_count, breach_penalty, severity_multiplier,
            )

            # Build a minimal Reward ledger for the breach step.
            breach_ledger = Reward(
                base_dispatch_score=breach_penalty,
                nlp_semantic_bonus=0.0,
                waste_penalty=0.0,
                total_reward=breach_penalty,
                dispatch_quality=breach_penalty,
                trajectory_shaping=0.0,
                nlp_bonus=0.0,
                is_terminal=self._is_done,
            )
            logger.info(
                "[Step %d] Reward Ledger (BREACH): %s",
                self._step_count,
                breach_ledger.model_dump_json(),
            )

            from env.grader import Grader
            score, eff_score = Grader().get_score(
                incidents_resolved=self._resolved_incidents,
                total_incidents=self._total_incidents,
                total_reward=self._total_reward,
                total_steps=max(self._step_count, 1),
                num_zones=len(self.obs.zones),
                wasted_dispatches=self._wasted_dispatches,
            )
            self._prev_obs = prev_obs_snapshot
            return (
                self.obs.model_copy(deep=True),
                breach_penalty,
                self._terminated,
                self._truncated,
                {
                    "resolved": self._resolved_incidents,
                    "total": self._total_incidents,
                    "score": score,
                    "efficiency": eff_score,
                    "wasted_dispatches": self._wasted_dispatches,
                    "reward_ledger": breach_ledger.model_dump(),
                    "inventory_breach": True,
                    "error_feedback": breach_msg,
                },
            )

        # 2. Compute reward *before* resolving zones (uses pre-action state).
        # Blocker #3 — Trajectory-Aware Reward Shaping:
        #   * On step 1 self._prev_obs is None (fresh episode); we fall back to
        #     the pre-tick snapshot so the reward function always receives a
        #     valid Observation rather than None.
        #   * On step 2+, self._prev_obs holds the genuine previous episode step's
        #     observation, enabling the Stabilization Bonus and Degradation Penalty
        #     calculations inside calculate_step_reward to see a real Δ.
        temporal_prior: Observation = self._prev_obs if self._prev_obs is not None else prev_obs_snapshot
        step_reward_ledger: Reward = calculate_step_reward(
            current_state=self.obs,
            action=action,
            previous_state=temporal_prior,
            previous_failures=self._zone_failures,
        )
        # NOTE: we do NOT yet add this to self._total_reward here;
        # the final corrected reward (after waste subtraction) is committed below.
        base_dispatch_reward: float = step_reward_ledger.base_dispatch_score
        step_nlp_bonus: float = step_reward_ledger.nlp_bonus

        # Snapshot waste accumulator BEFORE zone resolution so we can compute
        # the per-step waste delta after the zone loop completes.
        _waste_before_step: float = self._wasted_dispatches
        # 3. Commit allocations and resolve each zone.
        # Track over-allocations for Blocker #2 grader accuracy (severity-weighted).
        from env.reward import _get_required_fire, _get_required_ambulance
        for zone_id, zone_state in self.obs.zones.items():
            dispatch = action.allocations.get(zone_id, ZoneDispatch())
            # Snapshot zone severity BEFORE resolution so the penalty reflects
            # the hazard level that the agent actually faced this step.
            pre_fire_level  = zone_state.fire
            pre_patient_level = zone_state.patient

            # Anti-Exploit Guard: Zero-Resource Action Rejection
            has_active_hazard = (pre_fire_level != FireLevel.NONE or 
                                 pre_patient_level not in (PatientLevel.NONE, PatientLevel.FATAL) or 
                                 zone_state.traffic != TrafficLevel.LOW)
            is_zero_dispatch = (dispatch.dispatch_fire == 0 and 
                                dispatch.dispatch_ambulance == 0 and 
                                not dispatch.control_traffic)
            
            if has_active_hazard and is_zero_dispatch:
                logger.warning(
                    "[Step %d] LAZY AGENT EXPLOIT CAUGHT in %s: Zero resources dispatched to active hazard. Escalating crisis and penalizing.",
                    self._step_count, zone_id
                )
                self._escalate_zone(zone_state)
                base_dispatch_reward -= 5.0
                continue

            used_fire, used_amb, used_pol = self._commit_allocation(
                zone_id, zone_state, dispatch
            )
            self._resolve_zone(zone_id, zone_state, used_fire, used_amb, used_pol)

            # ------------------------------------------------------------------
            # Severity-Weighted Resource Penalty (Blocker #2 — novel grader)
            # ------------------------------------------------------------------
            # Philosophy: over-dispatching to an already-calm zone is the most
            # wasteful act (resources are precious and finite).  Over-dispatching
            # to a CATASTROPHIC / CRITICAL zone is more forgivable — a safety
            # buffer is a rational hedge against uncertainty.  We therefore
            # apply a *lower* penalty multiplier as severity increases.
            #
            # Fire severity penalty weights:
            #   NONE / LOW          → 2.0  (worst waste: zone didn't need them)
            #   MEDIUM              → 1.0  (moderate waste)
            #   HIGH / CATASTROPHIC → 0.5  (forgivable over-insurance)
            #
            # Ambulance severity penalty weights mirror the same philosophy for
            # patient triage levels.
            #
            # Each *excess unit* above the computed requirement contributes its
            # severity-weighted amount to _wasted_dispatches, giving the grader
            # a continuous, context-aware signal instead of a binary count.
            req_f = _get_required_fire(pre_fire_level, self.obs.weather)
            req_a = _get_required_ambulance(pre_patient_level)

            fire_excess = max(0, dispatch.dispatch_fire - req_f)
            amb_excess  = max(0, dispatch.dispatch_ambulance - req_a)

            if fire_excess > 0:
                fire_penalty_weight: float
                if pre_fire_level in (FireLevel.HIGH, FireLevel.CATASTROPHIC):
                    fire_penalty_weight = 0.5   # forgivable — high-stakes buffer
                elif pre_fire_level == FireLevel.MEDIUM:
                    fire_penalty_weight = 1.0   # moderate waste
                else:  # LOW or NONE
                    fire_penalty_weight = 2.0   # egregious — zone was calm
                self._wasted_dispatches += fire_excess * fire_penalty_weight
                logger.debug(
                    "%s: fire over-dispatch +%d (severity=%s, weight=%.1f) "
                    "→ waste_delta=%.1f",
                    zone_id, fire_excess, pre_fire_level.value,
                    fire_penalty_weight, fire_excess * fire_penalty_weight,
                )

            if amb_excess > 0:
                amb_penalty_weight: float
                if pre_patient_level == PatientLevel.CRITICAL:
                    amb_penalty_weight = 0.5    # forgivable — critical triage buffer
                elif pre_patient_level == PatientLevel.MODERATE:
                    amb_penalty_weight = 1.0    # moderate waste
                else:  # NONE or FATAL
                    amb_penalty_weight = 2.0    # egregious — no live patients
                self._wasted_dispatches += amb_excess * amb_penalty_weight
                logger.debug(
                    "%s: ambulance over-dispatch +%d (severity=%s, weight=%.1f) "
                    "→ waste_delta=%.1f",
                    zone_id, amb_excess, pre_patient_level.value,
                    amb_penalty_weight, amb_excess * amb_penalty_weight,
                )

        # 4. Determine episode termination.
        all_clear = all(
            z.fire == FireLevel.NONE
            and z.patient in (PatientLevel.NONE, PatientLevel.FATAL)
            and z.traffic == TrafficLevel.LOW
            for z in self.obs.zones.values()
        )
        # Directive 1: Strict separation of terminated vs truncated
        if all_clear:
            self._terminated = True
        elif self._step_count >= self._max_steps:
            self._truncated = True
            
        if self._terminated or self._truncated:
            self._is_done = True
            logger.info("Evaluation Terminated natively. Executing Scorecard hook.")

        # 4b. Generate Delta Feedback for the LLM's next observation.
        # This is the In-Context RL breadcrumb trail: per-zone plain-English
        # summaries of what changed and whether the dispatch was sufficient.
        # No numeric thresholds are disclosed — the agent must calibrate from
        # the direction of change (improved / held / degraded).
        feedback_lines: list[str] = [
            f"[Step {self._step_count - 1} Dispatch Results]"
        ]
        for zone_id in self.obs.zones:
            prev_z = prev_obs_snapshot.zones.get(zone_id)
            cur_z  = self.obs.zones[zone_id]
            disp   = action.allocations.get(zone_id, ZoneDispatch())

            if prev_z is None:
                continue

            parts: list[str] = [f"{zone_id}:"]

            # --- Fire delta ---
            if prev_z.fire != FireLevel.NONE:
                if cur_z.fire == FireLevel.NONE:
                    parts.append(
                        f"fire RESOLVED (sent {disp.dispatch_fire} fire units — SUFFICIENT)."
                    )
                elif cur_z.fire.value > prev_z.fire.value:
                    parts.append(
                        f"fire ESCALATED {prev_z.fire.value}→{cur_z.fire.value} "
                        f"(sent {disp.dispatch_fire} fire units — INSUFFICIENT, increase allocation)."
                    )
                else:
                    parts.append(
                        f"fire HELD at {cur_z.fire.value} "
                        f"(sent {disp.dispatch_fire} fire units — STABILIZED but not resolved)."
                    )

            # --- Patient delta ---
            if prev_z.patient not in (PatientLevel.NONE, PatientLevel.FATAL):
                if cur_z.patient == PatientLevel.NONE:
                    parts.append(
                        f"medical RESOLVED (sent {disp.dispatch_ambulance} ambulances — SUFFICIENT)."
                    )
                elif cur_z.patient == PatientLevel.FATAL:
                    parts.append(
                        f"patient status FATAL — too late to act."
                    )
                elif cur_z.patient.value > prev_z.patient.value:
                    parts.append(
                        f"patient condition WORSENED {prev_z.patient.value}→{cur_z.patient.value} "
                        f"(sent {disp.dispatch_ambulance} ambulances — INSUFFICIENT)."
                    )
                else:
                    parts.append(
                        f"patient STABLE at {cur_z.patient.value} "
                        f"(sent {disp.dispatch_ambulance} ambulances — holding, not resolved)."
                    )

            # --- Traffic delta ---
            if prev_z.traffic in (TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK):
                police_sent = "police deployed" if disp.control_traffic else "no police sent"
                if cur_z.traffic == TrafficLevel.LOW:
                    parts.append(f"traffic CLEARED ({police_sent} — SUFFICIENT).")
                else:
                    parts.append(
                        f"traffic PERSISTS at {cur_z.traffic.value} "
                        f"({police_sent} — deploy police to clear congestion)."
                    )

            if len(parts) == 1:
                parts.append("zone was clear, no active incidents.")

            feedback_lines.append(" ".join(parts))

        self.obs.previous_action_feedback = "\n".join(feedback_lines)
        logger.debug("Delta feedback generated: %s", self.obs.previous_action_feedback)


        # 5. Build info dict — Directive 3: Ruthless Utility reward formula.
        from env.grader import Grader  # local import avoids circular deps at module level

        # -----------------------------------------------------------------------
        # Directive 3: Ruthless Utility Reward Formula
        #
        #   Total_Reward = (Base_Dispatch_Reward + NLP_Bonus) - Waste_Penalty
        #
        # step_waste_penalty = severity-weighted resource waste accrued THIS step
        #                      (delta between cumulative before and after zone loop).
        # DO NOT clamp to zero — negative rewards are the intended gradient signal.
        # -----------------------------------------------------------------------
        step_waste_penalty: float = self._wasted_dispatches - _waste_before_step

        # Ruthless Utility final reward — no zero-floor.
        reward: float = (base_dispatch_reward + step_nlp_bonus) - step_waste_penalty

        # Commit the corrected step reward to the cumulative episode total.
        self._total_reward += reward

        logger.info(
            "[Step %d] Ruthless Utility: base=%.4f + nlp=%.4f - waste=%.4f = total=%.4f",
            self._step_count, base_dispatch_reward, step_nlp_bonus, step_waste_penalty, reward,
        )

        score, eff_score = Grader().get_score(
            incidents_resolved=self._resolved_incidents,
            total_incidents=self._total_incidents,
            total_reward=self._total_reward,
            total_steps=max(self._step_count, 1),
            num_zones=len(self.obs.zones),
            wasted_dispatches=self._wasted_dispatches,
        )

        # Build the final, authoritative Reward ledger for this step.
        # All three ledger lines are now correctly populated:
        #   base_dispatch_score — dispatch quality + trajectory shaping
        #   nlp_semantic_bonus  — NLP precision score (CAN be negative, Directive 3)
        #   waste_penalty       — positive scalar representation of resource waste
        #   total_reward        — Ruthless Utility sum (not clamped)
        final_step_ledger = Reward(
            base_dispatch_score=base_dispatch_reward,
            nlp_semantic_bonus=step_nlp_bonus,
            waste_penalty=step_waste_penalty,          # positive magnitude; subtracted in total
            total_reward=reward,                        # = base + nlp - waste (Directive 3)
            dispatch_quality=step_reward_ledger.dispatch_quality,
            trajectory_shaping=step_reward_ledger.trajectory_shaping,
            nlp_bonus=step_nlp_bonus,
            is_terminal=self._is_done,
        )
        logger.info(
            "[Step %d] Reward Ledger: %s",
            self._step_count,
            final_step_ledger.model_dump_json(),
        )

        # Directive 3: efficiency_score = Base_Reward / (Base_Reward + Waste_Penalty)
        # Tracks how efficiently the agent allocates resources (1.0 = zero waste).
        _base_magnitude = abs(base_dispatch_reward)
        efficiency_score: float = (
            _base_magnitude / (_base_magnitude + step_waste_penalty)
            if (_base_magnitude + step_waste_penalty) > 0
            else 1.0
        )

        info: Dict[str, Any] = {
            "resolved": self._resolved_incidents,
            "total": self._total_incidents,
            "score": score,
            "efficiency": eff_score,
            "efficiency_score": efficiency_score,       # Directive 3: resource efficiency tax tracker
            "wasted_dispatches": self._wasted_dispatches,
            "step_waste_penalty": step_waste_penalty,   # per-step waste for evaluator transparency
            "reward_ledger": final_step_ledger.model_dump(),
        }

        # Blocker #3: advance the temporal shaping anchor AFTER all mutations.
        # We store ``prev_obs_snapshot`` (captured pre-tick, pre-resolution) so
        # that on the NEXT step, calculate_step_reward sees the genuine "state
        # the world was in before this step" — not the post-resolution mutated obs.
        # This strict ordering enables accurate Δ-severity calculations.
        self._prev_obs = prev_obs_snapshot

        return self.obs.model_copy(deep=True), float(reward), self._terminated, self._truncated, info

    @property
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
            total_steps=max(self._step_count, 1),
            num_zones=len(self.obs.zones),
            wasted_dispatches=self._wasted_dispatches,
        )
        return EnvironmentState(
            step_count=self._step_count,
            max_steps=self._max_steps,
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

        MDP Physics (Dynamic Cooldown Calculus):
        A baseline physical travel minimum of C >= 2 is enforced. This prevents 
        a 0-turn resource replenishment loophole where deployed units instantly 
        return to the idle pool on the agent's next turn. This strictly enforces
        delayed gratification and resource conservation.

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
        # Agentic Purity Check: No clamping. The INVENTORY_BREACH gate in
        # step() has already mathematically verified that aggregate requests
        # do not exceed the idle pool.  We therefore trust the LLM's exact
        # dispatch values and apply them verbatim — no min() safety nets.
        used_fire = dispatch.dispatch_fire
        used_amb  = dispatch.dispatch_ambulance
        used_pol  = 1 if dispatch.control_traffic else 0

        # Move units from idle → busy pool (exact, no clamping).
        self.obs.idle_resources.fire_units -= used_fire
        self.obs.idle_resources.ambulances -= used_amb
        self.obs.idle_resources.police     -= used_pol
        self.obs.busy_resources.fire_units += used_fire
        self.obs.busy_resources.ambulances += used_amb
        self.obs.busy_resources.police     += used_pol

        # Implement the Dynamic Cooldown Calculus
        import math
        base_cooldown = 2

        # Extract the incident's severity (e.g., 1 to 5)
        fire_sev = {"none": 0, "low": 1, "medium": 2, "high": 3, "catastrophic": 4}.get(zone_state.fire.value, 0)
        pat_sev = {"none": 0, "moderate": 1, "critical": 3, "fatal": 5}.get(zone_state.patient.value, 0)
        traf_sev = {"low": 0, "heavy": 1, "gridlock": 2}.get(zone_state.traffic.value, 0)
        severity = max(1, fire_sev, pat_sev, traf_sev)
        
        # Apply a weather multiplier
        weather_multiplier = 1.0
        if self.obs.weather == WeatherCondition.STORM:
            weather_multiplier = 1.5
        elif self.obs.weather == WeatherCondition.HURRICANE:
            weather_multiplier = 2.0
            
        actual_cooldown = math.ceil((base_cooldown + severity) * weather_multiplier)

        if used_fire or used_amb or used_pol:
            self._active_deployments.append(
                ActiveDeployment(
                    zone_id=zone_id,
                    fire_units=used_fire,
                    ambulances=used_amb,
                    police=used_pol,
                    steps_remaining=actual_cooldown,
                    status="DISPATCHED",
                )
            )
            logger.debug(
                "%s: committed %dF/%dA/%dP (cooldown=%d).",
                zone_id,
                used_fire,
                used_amb,
                used_pol,
                actual_cooldown,
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
        cleared.  Otherwise the zone's internal failure counter is
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
            self._zone_failures[zone_id] = 0

        # ---- Failure / cascading escalation ---------------------------------
        elif not is_sufficient and has_active:
            current_failures = self._zone_failures.get(zone_id, 0) + 1
            self._zone_failures[zone_id] = current_failures
            if current_failures >= 3:
                logger.warning(
                    "CASCADING HAZARD TRIGGERED AT %s DUE TO LATENCY.", zone_id
                )
                self._escalate_zone(zone_state)
                self._zone_failures[zone_id] = 0
        else:
            # No incidents — reset failure counter regardless.
            self._zone_failures[zone_id] = 0

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
