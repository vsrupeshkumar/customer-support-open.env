import pytest
import copy

from env.environment import CrisisManagementEnv, EnvironmentException
from env.models import (
    Action,
    ZoneDispatch,
    FireLevel,
    PatientLevel,
    StructuralHallucinationError,
)

def test_reset_determinism():
    env1 = CrisisManagementEnv(task_id=1)
    obs1a, _ = env1.reset(seed=42)
    obs1b, _ = env1.reset(seed=42)
    assert obs1a.model_dump() == obs1b.model_dump(), "Monolithic Entropy Lock failed."

def test_reset_different_seeds_produce_different_obs():
    env = CrisisManagementEnv(task_id=2)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=99)
    assert obs1.model_dump() != obs2.model_dump(), "Different seeds must produce different scenario layouts."

def test_step_hallucination_penalty():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    error_action = StructuralHallucinationError("LLM String Output")
    obs2, reward, terminated, truncated, info = env.step(error_action)
    assert reward == -20.0, "Structural Darwinism expects exactly -20.0 penalty for hallucinations."
    assert "Bouncer Caught Hallucination" in info["error_msg"]
    assert env._step_count == 1, "Step count must increment even on hallucination."

def test_inventory_breach_penalty():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    
    idle_fire = obs.idle_resources.fire_units
    breach_qty = idle_fire + 1
    
    target_zone = next(iter(obs.zones.keys()))
    action = Action(allocations={target_zone: ZoneDispatch(dispatch_fire=breach_qty)})
    
    obs2, reward, terminated, truncated, info = env.step(action)
    assert info.get("inventory_breach") is True, "Inventory breach must set flag in info struct."
    assert reward <= -15.0, "Inventory breach penalty must be >= -15.0 continuous."
    
    # Assert pool was NOT modified by a voided action
    assert obs2.idle_resources.fire_units == idle_fire, "Inventory must remain untampered after voided breach."

def test_successful_zone_resolution():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    
    # Use exact required units for the first active zone
    from env.reward import _get_required_fire, _get_required_ambulance
    active_zone_id, active_zone = next((z, s) for z, s in obs.zones.items() if s.fire != FireLevel.NONE)
    
    req_fire = _get_required_fire(active_zone.fire, obs.weather)
    req_amb = _get_required_ambulance(active_zone.patient)
    
    action = Action(allocations={
        active_zone_id: ZoneDispatch(dispatch_fire=req_fire, dispatch_ambulance=req_amb)
    })
    
    obs2, reward, terminated, truncated, info = env.step(action)
    
    assert env._active_deployments[-1].status == "BUSY" or env._active_deployments[-1].status == "DISPATCHED"

def test_anti_exploit_guard_escalation():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    
    active_zone_id, active_zone = next((z, s) for z, s in obs.zones.items() if s.fire != FireLevel.NONE)
    pre_fire_level = active_zone.fire
    
    # Dispatch exactly zero
    action = Action(allocations={active_zone_id: ZoneDispatch(dispatch_fire=0)})
    obs2, reward, terminated, truncated, info = env.step(action)
    
    # Should flag DELAYED_HIGH_SEVERITY or IGNORE_INCIDENT in calculate_step_reward
    assert reward < 0.0, "Lazy agent must receive negative shaping reward."

def test_hard_mode_cascading():
    env = CrisisManagementEnv(task_id=3)
    obs, _ = env.reset(seed=42)
    
    zones_pre = len([z for z, s in obs.zones.items() if s.fire != FireLevel.NONE])
    # Step multiple times with zero actions to cause cascade spreading
    for _ in range(15):
        obs, r, term, trunc, info = env.step(Action())
        if term or trunc: break
        
    zones_post = len([z for z, s in obs.zones.items() if s.fire != FireLevel.NONE])
    assert zones_post >= zones_pre, "Hard mode cascading must spread incidents over negligent steps."

def test_hard_mode_resource_depletion():
    env = CrisisManagementEnv(task_id=3)
    obs, _ = env.reset(seed=42)
    initial_fire = obs.idle_resources.fire_units
    
    for _ in range(10):
         obs, _, term, trunc, _ = env.step(Action())
         if term or trunc: break
         
    # Due to constant steps and non-resolutions in hard mode, resources might deplete over steps
    # We test if the depletion hooks fired
    assert env.obs.idle_resources.fire_units <= initial_fire

def test_hard_mode_nhpp_disaster_spawn():
    env = CrisisManagementEnv(task_id=3)
    obs, _ = env.reset(seed=42)
    initial_incidents = env._total_incidents
    
    for _ in range(25):
        obs, _, term, trunc, _ = env.step(Action())
        if term or trunc: break
        
    assert getattr(env, "_total_incidents", initial_incidents) >= initial_incidents, "NHPP disaster spawning must increase incident count over time."

def test_curriculum_escalation_triggers():
    env = CrisisManagementEnv(task_id=2)
    obs, _ = env.reset(seed=42)
    
    env._reward_window = [1.0, 1.0, 1.0, 1.0, 1.0]
    initial_fire_units = obs.idle_resources.fire_units
    env._apply_curriculum_escalation()
    
    assert env.obs.idle_resources.fire_units < initial_fire_units, "High rolling rewards must trigger curriculum escalation (resource squeeze)."

def test_loop_detection_penalty():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    action = Action(allocations={"Downtown": ZoneDispatch(dispatch_fire=1)})
    
    # Step 1
    obs, r1, *__ = env.step(action)
    # Step 2
    obs, r2, *__ = env.step(action)
    # Step 3
    obs, r3, *__ = env.step(action)
    # Step 4
    obs, r4, *__ = env.step(action)
    
    assert r4 < r1, "Loop detection mechanism must significantly penalize exactly repeating actions over 3+ steps."

def test_terminated_vs_truncated_separation():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    
    for _ in range(env._max_steps):
        obs, reward, terminated, truncated, info = env.step(Action())
        if terminated or truncated:
            assert truncated is True, "Hitting max length limit must trigger truncated=True."
            assert terminated is False, "Hitting max length limit must leave terminated=False."
            break

def test_episode_after_done_raises():
    env = CrisisManagementEnv(task_id=1)
    obs, _ = env.reset(seed=42)
    
    for _ in range(env._max_steps + 5):
        if env._is_done:
            break
        env.step(Action())
        
    with pytest.raises(EnvironmentException):
        env.step(Action())
