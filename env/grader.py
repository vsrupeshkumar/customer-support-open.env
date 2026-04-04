"""
Enterprise Grade Bounded Evaluation Engine and Scoring Metrics Calculator.
Designed for high-performance extraction of Reinforcement Learning POMDP metrics.
"""

import math
import logging
from typing import Tuple, Dict, Any, List

# Setup Telemetry Logging for Grader Metrics
logger = logging.getLogger("crisis_env.grader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - GRADER - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class GraderException(Exception):
    """Custom exception raised when normalization bounds are mathematically violated."""
    pass

class GraderConfig:
    """Singleton Configuration Matrix for Metric Evaluators."""
    WEIGHT_SUCCESS_RATE: float = 0.50
    WEIGHT_EFFICIENCY: float = 0.50
    BASE_OPTIMAL_INCIDENT_REWARD: float = 15.0
    THEORETICAL_LOWER_BOUND_PENALTY: float = 20.0
    MIN_EFFICIENCY_CLIP: float = 0.0
    MAX_EFFICIENCY_CLIP: float = 1.0

class MetricTracker:
    """Object tracking the rolling array metrics of the orchestration evaluation."""
    def __init__(self):
        self.recorded_scores: List[float] = []
        self.recorded_efficiencies: List[float] = []

    def record(self, score: float, efficiency: float) -> None:
        self.recorded_scores.append(score)
        self.recorded_efficiencies.append(efficiency)

    def compute_variance(self) -> float:
        """Computes statistical variance across evaluated efficiency matrices."""
        if not self.recorded_efficiencies: return 0.0
        mean_eff = sum(self.recorded_efficiencies) / len(self.recorded_efficiencies)
        return sum((x - mean_eff)**2 for x in self.recorded_efficiencies) / len(self.recorded_efficiencies)

class Grader:
    """
    Advanced POMDP Evaluation Matrix Calculator.
    Binds raw action rewards to normalized sigmoid limits.
    """
    def __init__(self):
        self.config = GraderConfig()
        self.tracker = MetricTracker()
        logger.info("Grader initialized with multi-variable normalization matrices.")

    def _calculate_success_rate(self, incidents_resolved: int, total_incidents: int) -> float:
        """Mathematically generates the normalized success mapping."""
        if total_incidents == 0:
            logger.debug("No incidents to map. Resolving absolute 1.0 success factor.")
            return 1.0
        sr = float(incidents_resolved) / float(total_incidents)
        logger.debug(f"Computed Raw Success Rate: {sr:.3f}")
        return sr

    def _calculate_efficiency(self, total_reward: float, total_incidents: int) -> float:
        """
        Computes bounded evaluation matrices preventing simple positive exploitation.
        Optimizes based on absolute theoretical payout structures.
        """
        if total_incidents == 0:
            return 1.0 
            
        optimal_theoretical = total_incidents * self.config.BASE_OPTIMAL_INCIDENT_REWARD
        shifted_ceiling = optimal_theoretical + (total_incidents * self.config.THEORETICAL_LOWER_BOUND_PENALTY)
        shifted_reward = total_reward + (total_incidents * self.config.THEORETICAL_LOWER_BOUND_PENALTY)
        
        if shifted_ceiling > 0:
            raw_eff = shifted_reward / shifted_ceiling
            bound_eff = max(self.config.MIN_EFFICIENCY_CLIP, min(self.config.MAX_EFFICIENCY_CLIP, raw_eff))
            logger.debug(f"Computed Efficiency Bounded Value: {bound_eff:.3f}")
            return bound_eff
        
        return 1.0
        
    def get_score(self, incidents_resolved: int, total_incidents: int, total_reward: float) -> Tuple[float, float]:
        """
        Capstone advanced grading primary hook:
        Returns calculated evaluating arrays: (success_rate_score, efficiency_score)
        
        Args:
            incidents_resolved (int): Quantified resolved hazards.
            total_incidents (int): Initial baseline hazards recorded.
            total_reward (float): Cumulative reward array gathered.
            
        Returns:
            Tuple[float, float]: Final Score, Final Efficiency normalized to [0.0, 1.0].
        """
        try:
            logger.debug("Ingesting step evaluation heuristics...")
            sr_score = self._calculate_success_rate(incidents_resolved, total_incidents)
            efficiency = self._calculate_efficiency(total_reward, total_incidents)
            
            # Combine metrics into a strictly bounded final multi-axis matrix representation
            final_score = (sr_score * self.config.WEIGHT_SUCCESS_RATE) + (efficiency * self.config.WEIGHT_EFFICIENCY)
            
            # Log telemetry
            self.tracker.record(final_score, efficiency)
            logger.debug(f"Final Score Vector Compiled -> Score: {final_score:.4f}, Efficiency: {efficiency:.4f}")
            
            return final_score, efficiency
            
        except Exception as e:
            logger.error(f"Critical mapping failure in evaluation matrix: {str(e)}")
            raise GraderException(f"Failed to compile grader arrays: {str(e)}")
