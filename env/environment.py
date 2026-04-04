"""
Robust Global Simulation Framework Engine.
Processes the discrete operational time steps of the POMDP orchestrator tracking cooldown arrays locally.
"""

import copy
import logging
from typing import Tuple, Dict, Any, List, Optional

from env.models import Observation, Action, EnvironmentState, FireLevel, PatientLevel, TrafficLevel, WeatherCondition, ActiveDeployment, ZoneDispatch
from env.tasks import create_task
from env.reward import compute_reward
from env.grader import Grader

# Logger setup
logger = logging.getLogger("crisis_env.engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] ENGINE - %(message)s'))
    logger.addHandler(ch)

class EnvironmentException(Exception):
    """Exception triggered on catastrophic simulation initialization bounding failures."""
    pass

class LifecycleManager:
    """Manages the instantiation limits for active deployments in memory dynamically."""
    
    @staticmethod
    def tick_deployments(obs: Observation, active_deployments: List[ActiveDeployment]) -> List[ActiveDeployment]:
        """Iterates through deployed unit frames mapping cooldown returns recursively."""
        new_deployments = []
        recovered_fire, recovered_amb, recovered_pol = 0, 0, 0
        
        for dep in active_deployments:
            dep.steps_remaining -= 1
            if dep.steps_remaining <= 0:
                # Synchronize pool recovery
                recovered_fire += dep.fire_units
                recovered_amb += dep.ambulances
                recovered_pol += dep.police
            else:
                new_deployments.append(dep)
                
        # Batch execution prevents race conditions across multi-zone arrays
        if recovered_fire > 0 or recovered_amb > 0 or recovered_pol > 0:
            logger.debug(f"Engine recovered {recovered_fire}F | {recovered_amb}A | {recovered_pol}P back to Idle pool.")
            obs.idle_resources.fire_units += recovered_fire
            obs.idle_resources.ambulances += recovered_amb
            obs.idle_resources.police += recovered_pol
            
            obs.busy_resources.fire_units -= recovered_fire
            obs.busy_resources.ambulances -= recovered_amb
            obs.busy_resources.police -= recovered_pol
            
        return new_deployments


class CrisisManagementEnv:
    """
    Main Entrypoint Context object for OpenAI compliance frameworks.
    Simulates high-end geographical constraints over time series dimensions.
    """
    def __init__(self, task_id: int = 1, seed: Optional[int] = None):
        """
        Initializes the Smart City Simulator engine context constraints.
        
        Args:
            task_id (int): Difficulty map selection parameter.
            seed (Optional[int]): Randomization state initializer tracking.
        """
        self.task_id = task_id
        
        try:
            self._task = create_task(task_id)
        except Exception as e:
            logger.error(f"Failed to mount task id {task_id}: {str(e)}")
            raise EnvironmentException(f"Task generation sequence failed optimally: {str(e)}")
            
        self.grader = Grader()
        self.active_deployments: List[ActiveDeployment] = []
        self.reset()
        logger.info(f"CrisisManagementEnv successfully booted locally against Task {task_id}.")
    
    def reset(self) -> Observation:
        """
        Completely flushes dynamic physics simulation states rendering initial evaluation variables.
        
        Returns:
            Observation: Absolute initial mapped properties.
        """
        logger.debug("Executing engine reset parameters...")
        self.obs = self._task.generate_initial_observation()
        self.total_reward = 0.0
        self.is_done = False
        self.active_deployments = []
        
        self.total_incidents = 0
        for z_name, z in self.obs.zones.items():
            if z.fire != FireLevel.NONE: self.total_incidents += 1
            if z.patient != PatientLevel.NONE: self.total_incidents += 1
            if z.traffic in [TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK]: self.total_incidents += 1
            
        self.resolved_incidents = 0
        self.lives_saved = 0
        return self.obs.model_copy(deep=True)


    def _commit_resource_allocation(self, zone_id: str, zone_state: Any, dispatch: ZoneDispatch) -> Tuple[int, int, int]:
        """Bounds dispatch mappings rigorously against idle queues preventing arbitrary hallucination execution."""
        used_fire = min(dispatch.dispatch_fire, self.obs.idle_resources.fire_units)
        used_amb = min(dispatch.dispatch_ambulance, self.obs.idle_resources.ambulances)
        used_pol = 1 if (dispatch.control_traffic and self.obs.idle_resources.police > 0) else 0

        self.obs.idle_resources.fire_units -= used_fire
        self.obs.idle_resources.ambulances -= used_amb
        self.obs.idle_resources.police -= used_pol

        self.obs.busy_resources.fire_units += used_fire
        self.obs.busy_resources.ambulances += used_amb
        self.obs.busy_resources.police += used_pol

        cooldown = 1
        if self.obs.weather == WeatherCondition.HURRICANE: cooldown = 3
        elif self.obs.weather == WeatherCondition.STORM: cooldown = 2
        if zone_state.traffic == TrafficLevel.GRIDLOCK: cooldown += 2
        
        if used_fire > 0 or used_amb > 0 or used_pol > 0:
            logger.debug(f"{zone_id} deployed {used_fire}F|{used_amb}A|{used_pol}P recursively -> locked out for {cooldown} vectors.")
            self.active_deployments.append(ActiveDeployment(zone_id=zone_id, fire_units=used_fire, ambulances=used_amb, police=used_pol, steps_remaining=cooldown))
            
        return used_fire, used_amb, used_pol


    def _resolve_zone_engine(self, zone_id: str, zone_state: Any, used_fire: int, used_amb: int, used_pol: int):
        """Processes the micro-events within an independent geographic sector based on input bounds."""
        from env.reward import _get_required_fire, _get_required_ambulance
        
        req_fire = _get_required_fire(zone_state.fire, self.obs.weather)
        req_amb = _get_required_ambulance(zone_state.patient)
        
        resolved_this_zone = False
        is_sufficient = True
        
        if req_fire > 0 and used_fire < req_fire: 
            is_sufficient = False
        if req_amb > 0:
            amb_mod = 2 if zone_state.traffic == TrafficLevel.GRIDLOCK and not used_pol else 0
            if used_amb < req_amb + amb_mod: is_sufficient = False
        if zone_state.traffic in [TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK] and not used_pol: 
            is_sufficient = False

        has_active_issues = (req_fire > 0 or req_amb > 0 or (zone_state.traffic in [TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK]))

        # Condition 1: Succesful resolution limits
        if is_sufficient and has_active_issues:
            resolved_this_zone = True
            if used_fire > 0 and zone_state.fire != FireLevel.NONE:
                zone_state.fire = FireLevel.NONE
                self.resolved_incidents += 1
            if used_amb > 0 and zone_state.patient not in [PatientLevel.NONE, PatientLevel.FATAL]:
                zone_state.patient = PatientLevel.NONE
                self.resolved_incidents += 1
                self.lives_saved += 18
            if used_pol > 0 and zone_state.traffic in [TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK]:
                zone_state.traffic = TrafficLevel.LOW
                self.resolved_incidents += 1

        # Condition 2: Cascading Micro-Failures Algorithm
        active_remaining = (zone_state.fire != FireLevel.NONE or zone_state.patient not in [PatientLevel.NONE, PatientLevel.FATAL])
        if not resolved_this_zone and active_remaining:
            zone_state.consecutive_failures += 1
            if zone_state.consecutive_failures >= 3:
                logger.warning(f"CASCADING HAZARD TRIGGERED AT {zone_id} DUE TO LATENCY.")
                if zone_state.fire == FireLevel.HIGH: zone_state.fire = FireLevel.CATASTROPHIC
                elif zone_state.fire == FireLevel.MEDIUM: zone_state.fire = FireLevel.HIGH
                elif zone_state.fire == FireLevel.LOW: zone_state.fire = FireLevel.MEDIUM
                
                if zone_state.patient == PatientLevel.CRITICAL: zone_state.patient = PatientLevel.FATAL
                elif zone_state.patient == PatientLevel.MODERATE: zone_state.patient = PatientLevel.CRITICAL
                
                if zone_state.traffic == TrafficLevel.HEAVY: zone_state.traffic = TrafficLevel.GRIDLOCK
                
                zone_state.consecutive_failures = 0
        else:
            zone_state.consecutive_failures = 0


    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Executes a single algorithmic step tracking evaluation mechanics against the network.
        
        Args:
            action (Action): Distributed dispatch array computed by the Agent network.
            
        Returns:
            Tuple containing:
            - Observation (Pydantic Mapped POMDP state)
            - Total Reward (float bound)
            - Done Flag (bool evaluation marker)
            - Info Dict (Logging tracking variables)
        """
        if self.is_done:
            raise EnvironmentException("Simulation actively terminated. Cannot dispatch further routing calculations over locked arrays.")
            
        self.obs.step += 1
        
        # Propagate time-series states 
        self.active_deployments = LifecycleManager.tick_deployments(self.obs, self.active_deployments)
        
        # Score computation matrices
        reward, _ = compute_reward(action, self.obs)
        self.total_reward += reward

        for zone_id, zone_state in self.obs.zones.items():
            dispatch = action.allocations.get(zone_id, ZoneDispatch())
            
            used_fire, used_amb, used_pol = self._commit_resource_allocation(zone_id, zone_state, dispatch)
            self._resolve_zone_engine(zone_id, zone_state, used_fire, used_amb, used_pol)

        # Check absolute completion array termination flags
        all_clear = True
        for z in self.obs.zones.values():
            if z.fire != FireLevel.NONE or z.patient not in [PatientLevel.NONE, PatientLevel.FATAL] or z.traffic != TrafficLevel.LOW:
                all_clear = False
                break
                
        if all_clear or self.obs.step >= self.obs.max_steps:
            self.is_done = True
            logger.info("Evaluation Terminated natively. Executing Scorecard hook.")
            
        score, eff_score = self.grader.get_score(self.resolved_incidents, self.total_incidents, self.total_reward)
        info = {"resolved": self.resolved_incidents, "total": self.total_incidents, "score": score, "efficiency": eff_score}
        
        return self.obs.model_copy(deep=True), float(reward), self.is_done, info

    def state(self) -> EnvironmentState:
        """Publishes the global meta-context structure instantly resolving deep matrices."""
        score, eff_score = self.grader.get_score(self.resolved_incidents, self.total_incidents, self.total_reward)
        return EnvironmentState(
            step_count=self.obs.step, max_steps=self.obs.max_steps, observation=self.obs.model_copy(deep=True),
            total_reward=self.total_reward, is_done=self.is_done, success=self.resolved_incidents == self.total_incidents,
            metrics={"eff": eff_score, "lives": self.lives_saved}
        )
