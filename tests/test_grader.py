import unittest

from env.grader import grade_episode
from env.grader import _compute_efficiency

class TestGrader(unittest.TestCase):

    def test_grade_perfect_score(self):
        # Perfect run: success=True (resolved=total), waste=0, optimal reward.
        # weights: 50% success, 30% efficiency, 20% resource
        
        # We manually structure an episode that hits maximum raw multipliers
        incidents_resolved = 10
        total_incidents = 10
        total_reward = 80.0    # 8.0 per incident max
        wasted_dispatches = 0.0 # 0 waste
        action_diversity = 0.5  # Greater than 0.3 threshold, so penalty is 1.0
        
        final_score = grade_episode(
            incidents_resolved=incidents_resolved,
            total_incidents=total_incidents,
            total_reward=total_reward,
            total_steps=10,
            num_zones=3,
            wasted_dispatches=wasted_dispatches,
            action_diversity=action_diversity
        )
        # 1.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2 = 1.0
        self.assertAlmostEqual(final_score, 1.0, places=3)

    def test_monotony_penalty(self):
        # Action diversity < 0.3 triggers penalty scaling.
        # Everything else perfect, so raw score = 1.0. 
        # multiplier = diversity / 0.3 = 0.15 / 0.3 = 0.5
        # Expected final score: 1.0 * 0.5 = 0.5
        
        final_score = grade_episode(
            incidents_resolved=10,
            total_incidents=10,
            total_reward=80.0,
            total_steps=10,
            num_zones=3,
            wasted_dispatches=0.0,
            action_diversity=0.15
        )
        
        self.assertAlmostEqual(final_score, 0.5, places=3)

    def test_efficiency_bounds(self):
        # Verify clamp logic in _compute_efficiency
        # Efficiency can never be lower than 0.0 or higher than 1.0.
        
        val_high = _compute_efficiency(total_reward=1000.0, total_incidents=1)
        self.assertAlmostEqual(val_high, 1.0) # Ceiled to 1.0

        val_low = _compute_efficiency(total_reward=-1000.0, total_incidents=1)
        self.assertAlmostEqual(val_low, 0.0) # Floored to 0.0

if __name__ == '__main__':
    unittest.main()
