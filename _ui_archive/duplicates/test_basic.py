import sys
from env.models import Observation, FireLevel, PatientLevel, TrafficLevel, ResourcePool, Action, WeatherCondition, ZoneState, ZoneDispatch
from env.reward import compute_reward
from env.environment import CrisisManagementEnv

def test_reward():
    zones = {
        "Downtown": ZoneState(fire=FireLevel.HIGH, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW),
        "Suburbs": ZoneState()
    }
    obs = Observation(
        weather=WeatherCondition.CLEAR,
        zones=zones,
        idle_resources=ResourcePool(fire_units=5, ambulances=0, police=0),
        busy_resources=ResourcePool()
    )
    
    act_correct = Action(allocations={"Downtown": ZoneDispatch(dispatch_fire=3)})
    r, is_res = compute_reward(act_correct, obs)
    assert r > 0.0, f"Expected positive, got {r}"
    assert is_res == True
    
    act_wrong = Action(allocations={"Downtown": ZoneDispatch(dispatch_fire=1)})
    r, is_res = compute_reward(act_wrong, obs)
    assert r < 0.0, f"Expected negative, got {r}"
    assert is_res == False
    
    act_delay = Action(allocations={})
    r, is_res = compute_reward(act_delay, obs)
    assert r < 0.0
    assert is_res == False
    print("✅ Spatial Node reward constraints verified")

def test_tasks_and_env():
    for t in [1, 2, 3]:
        env = CrisisManagementEnv(task_id=t)
        obs = env.reset()
        assert obs.zones is != {}
        
        action = Action(allocations={"Downtown": ZoneDispatch(dispatch_fire=1)})
        obs, reward, done, info = env.step(action)
        assert len(obs.zones) > 0
        assert type(reward) is float
        assert type(done) is bool
        print(f"✅ Spatial Task {t} initialized and stepped successfully")
        
def main():
    print("Running Multi-Node Spatial Capstone Tests...\n")
    test_reward()
    test_tasks_and_env()
    print("\nALL TESTS PASSED: Stunning Meta AI Hackathon Environment Ready.")

if __name__ == "__main__":
    main()
