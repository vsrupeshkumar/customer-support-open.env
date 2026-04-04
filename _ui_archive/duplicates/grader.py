"""
Advanced Grader for evaluating agent performance.

Provides nuanced scoring between 0.0-1.0 with:
- Partial credit for partially correct solutions
- Efficiency penalties for unnecessary steps
- Inappropriate escalation penalties
- Detailed scoring breakdown
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class GradeResult:
    """Result of grading an action."""
    score: float  # 0.0 to 1.0
    feedback: str
    is_correct: bool


class Grader:
    """
    Sophisticated grader for evaluating agent performance.
    
    Provides:
    - Deterministic, fair scoring by task
    - Partial credit for suboptimal but valid choices
    - Efficiency bonuses/penalties
    - Comprehensive feedback
    """
    
    def __init__(self):
        """Initialize grader with task score tracking."""
        self.task_scores: Dict[int, List[float]] = {0: [], 1: [], 2: []}
        self.task_details: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
    
    def grade_classification(self, action: str, expected: str) -> GradeResult:
        """
        Grade Task 0 (Easy) classification.
        
        Scoring:
        - Exact match: 1.0
        - Wrong: 0.0
        """
        action_norm = action.lower().strip()
        expected_norm = expected.lower().strip()
        
        if action_norm == expected_norm:
            return GradeResult(
                score=1.0,
                feedback=f"[OK] Correct classification: {action}",
                is_correct=True
            )
        else:
            return GradeResult(
                score=0.0,
                feedback=f"[FAIL] Incorrect. Expected: {expected}, Got: {action}",
                is_correct=False
            )
    
    def grade_action_selection(
        self,
        action: str,
        issue_severity: str,
        issue_type: str = None
    ) -> GradeResult:
        """
        Grade Task 1 (Medium) action selection with nuanced scoring.
        
        Scoring:
        - Optimal action for severity: 1.0
        - Valid but suboptimal: 0.5-0.7
        - Invalid action: -0.5
        """
        valid_actions = {"resolve", "ask_info", "escalate"}
        action_norm = action.lower().strip()
        
        # Check validity
        if action_norm not in valid_actions:
            return GradeResult(
                score=-0.5,
                feedback=f"[FAIL] Invalid action: {action}. Valid: {', '.join(valid_actions)}",
                is_correct=False
            )
        
        # Optimal actions by severity
        optimal_actions = {
            "low": "resolve",
            "medium": "ask_info",
            "high": "escalate"
        }
        
        expected = optimal_actions.get(issue_severity, "ask_info")
        
        if action_norm == expected:
            feedback = (
                f"[OK] Optimal action for {issue_severity} severity"
                + (f" {issue_type}" if issue_type else "")
            )
            return GradeResult(
                score=1.0,
                feedback=feedback,
                is_correct=True
            )
        else:
            # Partial credit for valid but suboptimal choices
            if issue_severity == "low":
                # For low severity, resolve is optimal
                if action_norm == "ask_info":
                    partial_score = 0.3
                    feedback = "[OK] Valid but slightly inefficient. Better: resolve"
                else:  # escalate
                    partial_score = 0.3
                    feedback = "[OK] Suboptimal escalation. Better: resolve"
            elif issue_severity == "medium":
                # For medium, ask_info is optimal
                partial_score = 0.5
                feedback = f"[OK] Valid but suboptimal for {issue_severity} severity. Better: {expected}"
            else:  # high severity
                # For high, escalate is optimal
                if action_norm == "ask_info":
                    partial_score = 0.4
                    feedback = "[OK] Too slow for high severity. Better: escalate"
                else:  # resolve
                    partial_score = 0.3
                    feedback = "[OK] Insufficient for high severity. Better: escalate"
            
            return GradeResult(
                score=partial_score,
                feedback=feedback,
                is_correct=False
            )
    
    def grade_interaction_step(
        self,
        action: str,
        current_sentiment: float,
        issue_severity: str,
        step_num: int,
        max_steps: int
    ) -> GradeResult:
        """
        Grade Task 2 (Hard) interaction step.
        
        Uses distributed partial credit scoring:
        - Issue understanding: +0.3 max
        - Good engagement: +0.4 max
        - Resolution progress: +0.3 max
        
        Penalties:
        - Unhelpful: -0.5
        - Unnecessary escalation: -0.2
        - Repeated mistakes: -0.3
        
        Final score capped at -0.5 to 1.0
        """
        if not action or not isinstance(action, str) or len(action.strip()) == 0:
            return GradeResult(
                score=-0.5,
                feedback="[FAIL] Empty or invalid response",
                is_correct=False
            )
        
        reward = 0.0
        feedback_parts = []
        action_lower = action.lower()
        
        # === UNDERSTANDING COMPONENT ===
        understanding_reward = 0.0
        
        understanding_words = ["understand", "appreciate", "concern", "sorry", "frustrat"]
        if any(word in action_lower for word in understanding_words):
            understanding_reward = 0.15
            feedback_parts.append("Understanding shown")
        
        # Severity-appropriate language
        if issue_severity == "high":
            urgent_words = ["immediately", "priority", "escalate", "urgent"]
            if any(word in action_lower for word in urgent_words):
                understanding_reward += 0.15
                feedback_parts.append("Urgency recognized")
        elif issue_severity == "low":
            casual_words = ["help", "process", "guide", "assist"]
            if any(word in action_lower for word in casual_words):
                understanding_reward += 0.10
                feedback_parts.append("Appropriate tone")
        else:  # medium
            balanced_words = ["help", "understand", "information", "details"]
            if any(word in action_lower for word in balanced_words):
                understanding_reward += 0.15
                feedback_parts.append("Balanced approach")
        
        reward += understanding_reward
        
        # === ACTION COMPONENT ===
        helpful_words = {
            "help": 0.08,
            "resolve": 0.10,
            "solution": 0.10,
            "assure": 0.08,
            "guarantee": 0.08,
            "ensure": 0.08
        }
        
        action_bonus = 0.0
        detected = []
        
        for word, points in helpful_words.items():
            if word in action_lower:
                action_bonus += points
                detected.append(word)
        
        action_bonus = min(action_bonus, 0.40)  # Cap at 0.40
        reward += action_bonus
        
        if detected:
            feedback_parts.append(f"Quality: {', '.join(set(detected[:2]))}")
        
        # === RESOLUTION COMPONENT ===
        resolution_bonus = 0.0
        action_words = ["refund", "replace", "processed", "resolved", "escalated"]
        
        if any(word in action_lower for word in action_words):
            resolution_bonus = 0.20
            feedback_parts.append("Action taken")
            
            # Efficiency bonus for early resolution
            if step_num <= 2:
                resolution_bonus += 0.10
                feedback_parts.append("Efficient")
        else:
            # Clarifying questions are okay in step 1
            clarify_words = ["could you", "can you", "details", "information"]
            if any(word in action_lower for word in clarify_words) and step_num <= 2:
                resolution_bonus = 0.10
                feedback_parts.append("Gathering info")
        
        reward += resolution_bonus
        
        # === PENALTIES ===
        # Harmful response
        harmful_words = ["refuse", "can't help", "policy", "denied"]
        if any(word in action_lower for word in harmful_words):
            penalty = -0.5
            reward += penalty
            feedback_parts.append("⚠ Unhelpful response")
        
        # Unnecessary escalation
        if "escalate" in action_lower and issue_severity == "low":
            penalty = -0.2
            reward += penalty
            feedback_parts.append("Unnecessary escalation")
        
        # Sentiment factor
        if current_sentiment > 0.75:
            satisfaction_bonus = 0.10
            reward += satisfaction_bonus
            feedback_parts.append("High satisfaction")
        elif current_sentiment < 0.3:
            # Already angry - need extra care
            if any(w in action_lower for w in ["sorry", "understand", "help"]):
                care_bonus = 0.05
                reward += care_bonus
                feedback_parts.append("Appropriate care")
        
        # Final score
        final_score = min(1.0, max(-0.5, reward))
        final_feedback = " | ".join(feedback_parts) if feedback_parts else "Minimal engagement"
        is_correct = final_score > 0.5
        
        return GradeResult(
            score=final_score,
            feedback=final_feedback,
            is_correct=is_correct
        )
    
    def record_score(self, task_id: int, score: float, details: Dict = None):
        """
        Record a score for a task.
        
        Args:
            task_id: 0, 1, or 2
            score: Score 0.0-1.0
            details: Optional metadata
        """
        if 0 <= task_id <= 2:
            self.task_scores[task_id].append(score)
            if details:
                self.task_details[task_id].append(details)
    
    def get_task_average(self, task_id: int) -> float:
        """
        Get average score for a specific task.
        Returns 0.0 if no scores recorded.
        """
        scores = self.task_scores.get(task_id, [])
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_overall_average(self) -> float:
        """
        Get overall average across all tasks.
        Returns 0.0 if no scores recorded.
        """
        all_scores = []
        for scores in self.task_scores.values():
            all_scores.extend(scores)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get complete scoring summary.
        
        Returns:
            Dictionary with task names and averages
        """
        task_names = [
            "Easy (Classification)",
            "Medium (Action)",
            "Hard (Interaction)"
        ]
        
        summary = {}
        
        for task_id in range(3):
            avg = self.get_task_average(task_id)
            summary[task_names[task_id]] = round(avg, 4)
        
        summary["Overall Average"] = round(self.get_overall_average(), 4)
        
        return summary
    
    def get_detailed_summary(self) -> str:
        """Get formatted summary string."""
        summary = self.get_summary()
        
        lines = ["SCORING SUMMARY", "=" * 50]
        
        for task_name, score in summary.items():
            if task_name != "Overall Average":
                status = "[OK]" if score >= 0.7 else "[OK]" if score >= 0.5 else "[FAIL]"
                lines.append(f"{status} {task_name:<25} {score:.4f}")
        
        lines.append("-" * 50)
        overall = summary["Overall Average"]
        lines.append(f"OVERALL: {overall:.4f}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all recorded scores."""
        self.task_scores = {0: [], 1: [], 2: []}
        self.task_details = {0: [], 1: [], 2: []}
