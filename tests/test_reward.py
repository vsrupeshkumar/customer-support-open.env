import unittest

from env.models import (
    FireLevel,
    PatientLevel,
    TrafficLevel,
    WeatherCondition,
    ZoneState,
    ZoneDispatch,
    Observation,
    ResourcePool,
    TaskLevel,
    Action
)
from env.reward import RewardConstants, _get_required_fire, _get_required_ambulance, calculate_step_reward

class TestRewardFunctions(unittest.TestCase):
    
    def test_required_fire(self):
        # Base Requirements Map: CATASTROPHIC=5, HIGH=3, MEDIUM=2, LOW=1, NONE=0
        self.assertEqual(_get_required_fire(FireLevel.NONE, WeatherCondition.CLEAR), 0)
        self.assertEqual(_get_required_fire(FireLevel.LOW, WeatherCondition.CLEAR), 1)
        self.assertEqual(_get_required_fire(FireLevel.CATASTROPHIC, WeatherCondition.CLEAR), 5)

        # Weather Friction
        self.assertEqual(_get_required_fire(FireLevel.NONE, WeatherCondition.HURRICANE), 0) # Friction only applies if fire exists
        self.assertEqual(_get_required_fire(FireLevel.LOW, WeatherCondition.STORM), 2) # 1 + 1
        self.assertEqual(_get_required_fire(FireLevel.CATASTROPHIC, WeatherCondition.HURRICANE), 7) # 5 + 2

    def test_required_ambulance(self):
        # Base Requirements Map: CRITICAL=3, MODERATE=1, FATAL=0, NONE=0
        self.assertEqual(_get_required_ambulance(PatientLevel.NONE), 0)
        self.assertEqual(_get_required_ambulance(PatientLevel.FATAL), 0)
        self.assertEqual(_get_required_ambulance(PatientLevel.MODERATE), 1)
        self.assertEqual(_get_required_ambulance(PatientLevel.CRITICAL), 3)

    def test_calculate_step_reward_ignore_penalty(self):
        """Testing double penalty avoidance logic: ensuring we only receive -4 and -5 logic."""
        zones = {
            "Downtown": ZoneState(fire=FireLevel.HIGH, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW)
        }
        obs = Observation(
            weather=WeatherCondition.CLEAR,
            zones=zones,
            idle_resources=ResourcePool(fire_units=10, ambulances=10, police=5),
            busy_resources=ResourcePool(fire_units=0, ambulances=0, police=0),
            task_level=TaskLevel.EASY
        )
        
        # Zero dispatch to a HIGH fire
        allocations = {
            "Downtown": ZoneDispatch(dispatch_fire=0, dispatch_ambulance=0, control_traffic=False)
        }
        action = Action(allocations=allocations)
        
        reward = calculate_step_reward(obs, action, obs) # Trajectory is flat
        
        # We expect:
        # IGNORE_INCIDENT = -4.0
        # DELAYED_HIGH_SEVERITY = -5.0
        # Total per-zone = -9.0
        # Waste Penalty = 0
        expected_base = RewardConstants.IGNORE_INCIDENT + RewardConstants.DELAYED_HIGH_SEVERITY
        
        self.assertEqual(reward.base_dispatch_score, expected_base)

    def test_calculate_step_reward_efficiency(self):
        """Test EFFICIENT_RESOLUTION (+1.0) and CORRECT_ALLOCATION (+2.0) bonuses."""
        zones = {
            "Downtown": ZoneState(fire=FireLevel.LOW, patient=PatientLevel.MODERATE, traffic=TrafficLevel.LOW)
        }
        obs = Observation(
            weather=WeatherCondition.CLEAR,
            zones=zones,
            idle_resources=ResourcePool(fire_units=10, ambulances=10, police=5),
            busy_resources=ResourcePool(fire_units=0, ambulances=0, police=0),
            task_level=TaskLevel.EASY
        )
        
        # Exact minimums: 1 fire, 1 amb
        allocations = {
            "Downtown": ZoneDispatch(dispatch_fire=1, dispatch_ambulance=1, control_traffic=False)
        }
        action = Action(allocations=allocations)
        
        reward = calculate_step_reward(obs, action, obs)
        
        expected_base = RewardConstants.CORRECT_ALLOCATION + RewardConstants.EFFICIENT_RESOLUTION
        
        self.assertEqual(reward.base_dispatch_score, expected_base)

if __name__ == '__main__':
    unittest.main()
