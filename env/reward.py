"""
Advanced Multi-Node Mathematical Reward Modulator.
Implements the core Markov Decision Process (MDP) reward payload equations for City-Wide Orchestrator.
Includes complex friction delays, gridlock scaling penalties, and optimal bounds detection.
"""

import math
import logging
from typing import Tuple, Dict, Any, Optional

from env.models import Action, Observation, FireLevel, PatientLevel, TrafficLevel, WeatherCondition, ZoneDispatch

# -----------------------------------------------------------------------------
# Module Telemetry & Config
# -----------------------------------------------------------------------------
logger = logging.getLogger("crisis_env.reward_evaluator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] REWARD MATH - %(message)s'))
    logger.addHandler(ch)

class RewardConstants:
    """Singleton pattern configuring global variables for bounded RL evaluation penalties."""
    BASE_OPTIMAL_PAYLOAD: float = 10.0
    EFFICIENCY_BONUS: float = 5.0
    WASTAGE_SCALAR: float = -2.0
    TRAFFIC_WASTAGE_SCALAR: float = -3.0
    INACTION_CRITICAL_PENALTY: float = -10.0
    INACTION_STANDARD_PENALTY: float = -5.0
    HURRICANE_BONUS_MULTIPLIER: float = 5.0
    WEATHER_HURRICANE_FIRE_FRICTION: int = 2
    WEATHER_STORM_FIRE_FRICTION: int = 1
    GRIDLOCK_AMB_FRICTION: int = 2

# -----------------------------------------------------------------------------
# Heuristic Boundary Calculators
# -----------------------------------------------------------------------------
def _get_required_fire(level: FireLevel, weather: WeatherCondition) -> int:
    """
    Computes the exact mathematical resource threshold required to contain a given Fire Severity.
    
    Args:
        level (FireLevel): The current operational hazard severity of the fire sector.
        weather (WeatherCondition): The ambient kinetic weather modifier (e.g. Storm, Hurricane).
        
    Returns:
        int: The absolute minimum number of Fire Units required to process a successful mitigation step.
    """
    logger.debug(f"Evaluating Fire Matrix for Level: {level.value} under Weather: {weather.value}")
    req = 0
    
    # Establish base severity requirements
    if level == FireLevel.CATASTROPHIC:
        req = 5
    elif level == FireLevel.HIGH:
        req = 3
    elif level == FireLevel.MEDIUM:
        req = 2
    elif level == FireLevel.LOW:
        req = 1
        
    # Kinetic weather friction dynamically elevates resource allocation requirements natively.
    if weather in [WeatherCondition.STORM, WeatherCondition.HURRICANE] and req > 0:
        modifier = RewardConstants.WEATHER_HURRICANE_FIRE_FRICTION if weather == WeatherCondition.HURRICANE else RewardConstants.WEATHER_STORM_FIRE_FRICTION
        req += modifier
        logger.debug(f"Weather {weather.value} escalated fire req by {modifier}.")
        
    return req

def _get_required_ambulance(level: PatientLevel) -> int:
    """
    Evaluates the triage bounds necessary to resolve medical casualties avoiding cascading fatal events.
    Notice: Gridlock friction operates exclusively in the deployment payload evaluation, not base requirements.
    
    Args:
        level (PatientLevel): The dynamic state of the medical casualty sector.
        
    Returns:
        int: The required mapping array of Ambulance dispatches.
    """
    logger.debug(f"Evaluating Medical Context boundaries: {level.value}")
    req = 0
    
    if level == PatientLevel.FATAL: 
        logger.warning("Agent encountered FATAL casualty. Reward limits strictly constrained.")
        return 0 # Triggers absolute penalty globally
        
    if level == PatientLevel.CRITICAL: 
        req = 3
    elif level == PatientLevel.MODERATE: 
        req = 1
        
    return req

# -----------------------------------------------------------------------------
# Main Evaluation Orchestrator
# -----------------------------------------------------------------------------

def evaluate_zone_dispatch(zone_id: str, zone_state: Any, dispatch: ZoneDispatch, obs: Observation) -> Tuple[float, bool]:
    """
    Isolates evaluating single-zone physics within the global routing map.
    """
    req_fire = _get_required_fire(zone_state.fire, obs.weather)
    req_amb = _get_required_ambulance(zone_state.patient)
    req_traffic = (zone_state.traffic in [TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK])
    
    has_incidents = (req_fire > 0 or req_amb > 0 or req_traffic)
    zone_reward = 0.0
    zone_resolved = True
    
    # Check 1: Wastage over an empty zone
    if not has_incidents:
        if dispatch.dispatch_fire > 0 or dispatch.dispatch_ambulance > 0:
            logger.debug(f"Wastage mapped over safe zone {zone_id}.")
            zone_reward -= 5.0
        return zone_reward, True
        
    # Check 2: Absolute paralysis / inaction
    is_action_empty = (dispatch.dispatch_fire == 0 and dispatch.dispatch_ambulance == 0 and not dispatch.control_traffic)
    if is_action_empty:
        penalty = RewardConstants.INACTION_CRITICAL_PENALTY if (zone_state.fire == FireLevel.CATASTROPHIC or zone_state.patient == PatientLevel.CRITICAL) else RewardConstants.INACTION_STANDARD_PENALTY
        logger.debug(f"Inaction triggered over active disaster zone {zone_id}. Yielding {penalty}.")
        return penalty, False

    # Check 3: Mathematical Wastage Penetration
    wastage_penalty = 0.0
    if dispatch.dispatch_fire > 0 and req_fire == 0: 
        wastage_penalty += RewardConstants.WASTAGE_SCALAR * dispatch.dispatch_fire
    if dispatch.dispatch_ambulance > 0 and req_amb == 0: 
        wastage_penalty += RewardConstants.WASTAGE_SCALAR * dispatch.dispatch_ambulance
    if dispatch.control_traffic and not req_traffic: 
        wastage_penalty += RewardConstants.TRAFFIC_WASTAGE_SCALAR
        
    if wastage_penalty < 0:
        # Heavily penalized bounded scalar calculation
        return RewardConstants.INACTION_STANDARD_PENALTY + wastage_penalty, False

    # Check 4: Dispatch Math Sufficiency
    is_sufficient = True
    if req_fire > 0 and dispatch.dispatch_fire < req_fire: 
        is_sufficient = False
    
    amb_modifier = 0
    if zone_state.traffic == TrafficLevel.GRIDLOCK and not dispatch.control_traffic:
        amb_modifier = RewardConstants.GRIDLOCK_AMB_FRICTION
        logger.debug(f"Gridlock cascading logic triggered in {zone_id} without Police dispatch.")
        
    if req_amb > 0 and (dispatch.dispatch_ambulance < (req_amb + amb_modifier)): 
        is_sufficient = False
        
    if req_traffic and not dispatch.control_traffic: 
        is_sufficient = False

    # Check 5: Success & Optimal Shaping
    if is_sufficient:
        efficiency_bonus = 0.0
        if dispatch.dispatch_fire == req_fire and dispatch.dispatch_ambulance == req_amb:
            efficiency_bonus = RewardConstants.EFFICIENCY_BONUS # Absolute precision multiplier 
        
        base_reward = RewardConstants.BASE_OPTIMAL_PAYLOAD
        if obs.weather == WeatherCondition.HURRICANE: 
            base_reward += RewardConstants.HURRICANE_BONUS_MULTIPLIER # Global complexity scalar
            
        zone_reward += (base_reward + efficiency_bonus)
        zone_resolved = True
    else:
        zone_reward += RewardConstants.INACTION_STANDARD_PENALTY
        zone_resolved = False
        
    return zone_reward, zone_resolved

def compute_reward(action: Action, obs: Observation) -> tuple[float, bool]:
    """
    Computes global multi-zone spatial rewards.
    Iterates dynamically combining single vectors locally.
    
    Args:
        action: Dispatched orchestration routing map.
        obs: Observation multi-array.
    """
    global_total_reward = 0.0
    global_all_resolved = True
    
    for zone_id, zone_state in obs.zones.items():
        dispatch: ZoneDispatch = action.allocations.get(zone_id, ZoneDispatch())
        
        z_reward, z_resolved = evaluate_zone_dispatch(zone_id, zone_state, dispatch, obs)
        global_total_reward += z_reward
        if not z_resolved:
            global_all_resolved = False
            
    return global_total_reward, global_all_resolved
