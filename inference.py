import argparse
import os
import json
import random
import textwrap
from typing import List, Optional, Dict, Any
from copy import deepcopy

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from env import CrisisManagementEnv
from env.models import Action, Observation, ZoneDispatch, FireLevel, PatientLevel, TrafficLevel, WeatherCondition

# ==========================================
# OpenEnv Required Logging Format Helpers
# ==========================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_think(step: int, critical: str, risk: str, strategy: str) -> None:
    print(f"[THINK] step={step} critical={critical} risk={risk} strategy={strategy}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float], efficiency: float = 0.95, hazards_prevented: int = 0, stability: float = 0.90) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} efficiency={efficiency:.2f} hazards_prevented={hazards_prevented} stability={stability:.2f} rewards={rewards_str}", flush=True)

# ==========================================
# AI Agent Implementation
# ==========================================

class StrategicAgent:
    def __init__(self):
        self.model = os.environ.get("MODEL_NAME", "strategic-heuristic-agent")
        self.api_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.history = []
        self.last_reward = 0.0
        self.last_action = None
        
    def _fire_reqs(self, level: FireLevel, weather: WeatherCondition) -> int:
        req = {"none": 0, "low": 1, "medium": 2, "high": 3, "catastrophic": 5}.get(level.value, 0)
        if req > 0:
            if weather == WeatherCondition.HURRICANE:
                req += 2
            elif weather == WeatherCondition.STORM:
                req += 1
        return req
        
    def _patient_reqs(self, level: PatientLevel) -> int:
        return {"none": 0, "moderate": 1, "critical": 3, "fatal": 0}.get(level.value, 0)

    def get_action(self, obs: Observation, step: int) -> tuple[Action, Optional[str]]:
        try:
            idle_fire = obs.idle_resources.fire_units
            idle_amb = obs.idle_resources.ambulances
            idle_pol = obs.idle_resources.police
            
            allocations = {zone: {"dispatch_fire": 0, "dispatch_ambulance": 0, "control_traffic": False} for zone in obs.zones.keys()}
            
            zone_scores = []
            for z_name, z_state in obs.zones.items():
                score = 0
                if z_state.fire == FireLevel.CATASTROPHIC: score += 100
                elif z_state.fire == FireLevel.HIGH: score += 50
                
                if z_state.patient == PatientLevel.CRITICAL: score += 80
                
                if z_state.traffic == TrafficLevel.GRIDLOCK: score += 40
                
                # Cascading failure adaptation
                score += z_state.consecutive_failures * 15
                
                if self.last_reward < 0 and self.last_action:
                    # Adaptive memory: increase priority if reward was negative recently
                    prev_alloc = self.last_action.allocations.get(z_name)
                    if prev_alloc and (prev_alloc.dispatch_fire > 0 or prev_alloc.dispatch_ambulance > 0):
                        score += 5 
                
                zone_scores.append((score, z_name, z_state))
                
            zone_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Step phase logic
            # Step 1-3: Aggressive
            # Step 4-8: Stabilize
            # Step 9+: Optimize
            req_multiplier = 1.0
            
            # 1. Fire (Priority 1)
            # Step 1: guarantee at least 1 to any active fire if possible
            for score, z_name, z_state in zone_scores:
                if z_state.fire != FireLevel.NONE and idle_fire > 0:
                    allocations[z_name]["dispatch_fire"] = 1
                    idle_fire -= 1
                    
            for score, z_name, z_state in zone_scores:
                req = int(self._fire_reqs(z_state.fire, obs.weather) * req_multiplier)
                if z_state.consecutive_failures > 0: req += 1
                
                # subtract the 1 we already gave
                if allocations[z_name]["dispatch_fire"] > 0:
                    req -= 1
                    
                disp = min(idle_fire, req)
                if disp > 0:
                    allocations[z_name]["dispatch_fire"] += disp
                    idle_fire -= disp

            # 2. Medical (Priority 2)
            for score, z_name, z_state in zone_scores:
                if z_state.patient != PatientLevel.NONE and idle_amb > 0:
                    allocations[z_name]["dispatch_ambulance"] = 1
                    idle_amb -= 1
                    
            for score, z_name, z_state in zone_scores:
                req = int(self._patient_reqs(z_state.patient) * req_multiplier)
                if z_state.consecutive_failures > 0: req += 1
                
                if allocations[z_name]["dispatch_ambulance"] > 0:
                    req -= 1
                    
                disp = min(idle_amb, req)
                if disp > 0:
                    allocations[z_name]["dispatch_ambulance"] += disp
                    idle_amb -= disp
                    
            # 3. Traffic (Priority 3)
            for score, z_name, z_state in zone_scores:
                needs_police = (z_state.traffic in [TrafficLevel.HEAVY, TrafficLevel.GRIDLOCK])
                if needs_police and idle_pol > 0:
                    allocations[z_name]["control_traffic"] = True
                    idle_pol -= 1
                    
            # Resource optimization (distribute remaining)
            for score, z_name, z_state in zone_scores:
                if idle_fire > 0 and z_state.fire != FireLevel.NONE:
                    allocations[z_name]["dispatch_fire"] += 1
                    idle_fire -= 1
                if idle_amb > 0 and z_state.patient != PatientLevel.NONE:
                    allocations[z_name]["dispatch_ambulance"] += 1
                    idle_amb -= 1
                    
            action = Action(allocations={k: ZoneDispatch(**v) for k, v in allocations.items()})
            self.last_action = action
            return action, None
        except Exception as e:
            return Action(allocations={}), str(e)


from metrics_tracker import MetricsTracker

def generate_reasoning(obs: Observation, action: Action, reward: float) -> tuple[str, str, str]:
    # Identify critical zone
    zone_scores = []
    for z_name, z_state in obs.zones.items():
        score = 0
        if z_state.fire == FireLevel.CATASTROPHIC: score += 100
        elif z_state.fire == FireLevel.HIGH: score += 50
        if z_state.patient == PatientLevel.CRITICAL: score += 80
        if z_state.traffic == TrafficLevel.GRIDLOCK: score += 40
        zone_scores.append((score, z_name))
    
    zone_scores.sort(reverse=True)
    critical = zone_scores[0][1] if zone_scores else "None"
    
    # Assess risk
    highest_score = zone_scores[0][0] if zone_scores else 0
    if highest_score > 80: risk = "High"
    elif highest_score > 40: risk = "Medium"
    else: risk = "Low"
    
    # Determine strategy
    if obs.step <= 3:
        strategy = "AggressiveContainment"
    elif obs.step <= 8:
        strategy = "StabilizeAndHeal"
    else:
        strategy = "OptimizeResources"
    return critical, risk, strategy

def run_agent(agent: StrategicAgent, task_id: int):
    env = CrisisManagementEnv(task_id=task_id)
    obs = env.reset()
    metrics = MetricsTracker()

    log_start(task=str(task_id), env="adaptive-crisis", model=agent.model)      

    rewards = []
    error = None
    step_count = 0
    final_score = 0.0
    success = False

    while not env.is_done:
        step_count += 1
        action, get_action_error = agent.get_action(obs, step_count)
        
        critical, risk, strategy = generate_reasoning(obs, action, agent.last_reward)
        log_think(step_count, critical, risk, strategy)

        # Make the action serializable cleanly on one line
        action_json_str = json.dumps(action.model_dump(mode='json'), separators=(',', ':'))

        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        agent.last_reward = reward
        metrics.update(reward, action, obs, done)

        # Use either action generation error or step execution error
        step_error = get_action_error

        log_step(
            step=step_count, 
            action=action_json_str, 
            reward=float(reward), 
            done=done, 
            error=step_error
        )
        
        if done:
            final_score = info.get("score", 0.0)
            # Define success as getting > 0.5 final score per the hackathon grading criteria 
            success = final_score >= 0.5
            break
            
    # Final metrics logging
    summary = metrics.get_summary()

    log_end(success=success, steps=step_count, score=final_score, rewards=rewards, 
            efficiency=summary["efficiency"], hazards_prevented=summary["hazards_prevented"], stability=summary["stability"])

if __name__ == "__main__":
    agent = StrategicAgent()
    # Run once per task as required to reproduce baseline scores
    for t_id in [1, 2, 3]:
        run_agent(agent, t_id)
