"""
AI Crisis Management - Task Definitions

4 Tasks with increasing difficulty:
- Task 0 (Easy):    Single crisis classification
- Task 1 (Medium):  Resource allocation under constraints
- Task 2 (Hard):    Multi-agent coordination
- Task 3 (Extreme): Cascading crisis with time pressure
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any


@dataclass
class Task:
    """Base task definition."""
    task_id: int
    name: str
    description: str
    difficulty: str
    max_steps: int
    num_agents: int


# ─────────────────────────────────────────────
#  TASK 0 — EASY: Crisis Classification
# ─────────────────────────────────────────────

class CrisisClassificationTask(Task):
    """
    Task 0 - EASY: Single Crisis Classification

    Agent sees a crisis report and must classify it into one of 5 categories:
    fire / accident / medical / flood / infrastructure

    Reward:
    - Correct:   +1.0
    - Incorrect:  0.0
    """

    def __init__(self):
        super().__init__(
            task_id=0,
            name="Crisis Classification",
            description="Classify incoming crisis report into correct category",
            difficulty="easy",
            max_steps=1,
            num_agents=1
        )
        self.crisis_reports = [
            {
                "report": "Massive fire spotted at City Mall, Zone A. Smoke visible from 2km.",
                "expected": "fire",
                "affected": 200,
                "severity": "high"
            },
            {
                "report": "Multi-vehicle collision on Highway 7. 3 cars involved, injuries reported.",
                "expected": "accident",
                "affected": 12,
                "severity": "medium"
            },
            {
                "report": "Patient in cardiac arrest at Sector 4 apartment. Immediate help needed.",
                "expected": "medical",
                "affected": 1,
                "severity": "critical"
            },
            {
                "report": "Flash flood warning in Zone C. Water levels rising rapidly.",
                "expected": "flood",
                "affected": 500,
                "severity": "high"
            },
            {
                "report": "Power grid failure across northern district. Hospital backup failing.",
                "expected": "infrastructure",
                "affected": 5000,
                "severity": "critical"
            },
            {
                "report": "Explosion and fire at chemical plant, Zone B. Evacuation required.",
                "expected": "fire",
                "affected": 150,
                "severity": "critical"
            },
            {
                "report": "Train derailment at Central Station. Multiple casualties feared.",
                "expected": "accident",
                "affected": 80,
                "severity": "critical"
            },
            {
                "report": "Mass food poisoning at school cafeteria. 40 students unwell.",
                "expected": "medical",
                "affected": 40,
                "severity": "high"
            },
            {
                "report": "River dam showing structural cracks. Downstream residents at risk.",
                "expected": "flood",
                "affected": 2000,
                "severity": "critical"
            },
            {
                "report": "Bridge collapse on Ring Road. Traffic blocked, people trapped.",
                "expected": "infrastructure",
                "affected": 75,
                "severity": "high"
            }
        ]

    def valid_actions(self) -> List[str]:
        return ["fire", "accident", "medical", "flood", "infrastructure"]

    def evaluate(self, action: str, expected: str) -> Tuple[float, str]:
        if action.lower().strip() == expected.lower():
            return 1.0, f"✅ Correct classification: {action}"
        return 0.0, f"❌ Wrong. Expected: {expected}, Got: {action}"


# ─────────────────────────────────────────────
#  TASK 1 — MEDIUM: Resource Allocation
# ─────────────────────────────────────────────

class ResourceAllocationTask(Task):
    """
    Task 1 - MEDIUM: Resource Allocation Under Constraints

    Agent sees crisis type + severity + available resources.
    Must choose optimal resource dispatch level:
    - send_minimum:  1-2 units (for low severity)
    - send_standard: 3-5 units (for medium severity)
    - send_maximum:  Full force (for high/critical)
    - evacuate:      Trigger full area evacuation

    Reward:
    - Optimal:   +1.0
    - Suboptimal: +0.3 to +0.7
    - Wrong:     -0.5
    """

    def __init__(self):
        super().__init__(
            task_id=1,
            name="Resource Allocation",
            description="Allocate resources optimally based on crisis severity",
            difficulty="medium",
            max_steps=2,
            num_agents=1
        )
        self.scenarios = [
            {
                "crisis_type": "fire",
                "severity": "low",
                "report": "Small bin fire in alley. No injuries.",
                "available": {"fire_truck": 5, "ambulance": 3},
                "expected": "send_minimum",
                "reason": "Minor fire, minimal resources needed"
            },
            {
                "crisis_type": "accident",
                "severity": "medium",
                "report": "Car crash, 2 injured. Road partially blocked.",
                "available": {"ambulance": 4, "police_car": 3},
                "expected": "send_standard",
                "reason": "Moderate accident needs standard response"
            },
            {
                "crisis_type": "medical",
                "severity": "critical",
                "report": "Building collapse. 20+ people trapped inside.",
                "available": {"ambulance": 6, "rescue_team": 4, "helicopter": 2},
                "expected": "send_maximum",
                "reason": "Critical mass casualty — full force required"
            },
            {
                "crisis_type": "flood",
                "severity": "critical",
                "report": "Dam burst. 2000 residents in flood path.",
                "available": {"rescue_team": 5, "helicopter": 3, "police_car": 8},
                "expected": "evacuate",
                "reason": "Mass flood threat requires full evacuation"
            },
            {
                "crisis_type": "fire",
                "severity": "high",
                "report": "Apartment block fire. 50 residents trapped.",
                "available": {"fire_truck": 6, "ambulance": 4, "rescue_team": 3},
                "expected": "send_maximum",
                "reason": "High-severity fire needs maximum response"
            },
            {
                "crisis_type": "infrastructure",
                "severity": "low",
                "report": "Streetlight malfunction on suburban road.",
                "available": {"police_car": 5},
                "expected": "send_minimum",
                "reason": "Minor infrastructure — minimal response"
            },
            {
                "crisis_type": "medical",
                "severity": "medium",
                "report": "Elderly person fell, possible fracture.",
                "available": {"ambulance": 3},
                "expected": "send_standard",
                "reason": "Standard medical response adequate"
            },
            {
                "crisis_type": "flood",
                "severity": "high",
                "report": "Flash flood in market area. 100 people stranded.",
                "available": {"rescue_team": 4, "helicopter": 2},
                "expected": "send_maximum",
                "reason": "High flood with many people — max response"
            }
        ]
        self.valid_action_list = ["send_minimum", "send_standard", "send_maximum", "evacuate"]

    def evaluate(self, action: str, severity: str, crisis_type: str) -> Tuple[float, str]:
        action = action.lower().strip()
        if action not in self.valid_action_list:
            return -0.5, f"❌ Invalid action: {action}"

        optimal = {
            "low": "send_minimum",
            "medium": "send_standard",
            "high": "send_maximum",
            "critical": "evacuate" if crisis_type in ["flood"] else "send_maximum"
        }
        expected = optimal.get(severity, "send_standard")

        if action == expected:
            return 1.0, f"✅ Optimal allocation for {severity} {crisis_type}"

        # Partial credit
        severity_order = ["send_minimum", "send_standard", "send_maximum", "evacuate"]
        action_idx = severity_order.index(action) if action in severity_order else -1
        expected_idx = severity_order.index(expected) if expected in severity_order else -1
        distance = abs(action_idx - expected_idx)

        if distance == 1:
            return 0.5, f"⚠️ Close but suboptimal. Better: {expected}"
        else:
            return 0.2, f"⚠️ Poor allocation for {severity} severity. Better: {expected}"


# ─────────────────────────────────────────────
#  TASK 2 — HARD: Multi-Agent Coordination
# ─────────────────────────────────────────────

class MultiAgentCoordinationTask(Task):
    """
    Task 2 - HARD: Multi-Agent Crisis Coordination

    3 simultaneous crises. 3 agents must coordinate:
    - Fire Commander   → manages fire response
    - Medical Dispatch → manages ambulances and hospitals
    - Traffic Control  → clears routes, manages evacuations

    Agents must coordinate actions. Conflicting decisions are penalized.
    Synergistic decisions are rewarded.

    Reward per step:
    - Correct role action:  +0.3
    - Coordination sync:    +0.4
    - Lives saved:          +0.3
    - Conflict penalty:     -0.4
    - Delay penalty:        -0.2
    """

    def __init__(self):
        super().__init__(
            task_id=2,
            name="Multi-Agent Coordination",
            description="3 AI agents coordinate to resolve simultaneous crises",
            difficulty="hard",
            max_steps=5,
            num_agents=3
        )
        self.scenarios = [
            {
                "id": "S1",
                "description": "Fire + injured civilians + blocked roads",
                "crises": [
                    {"type": "fire", "severity": "high", "location": "Zone A", "affected": 50},
                    {"type": "medical", "severity": "critical", "location": "Zone A", "affected": 8},
                    {"type": "accident", "severity": "medium", "location": "Highway 7", "affected": 12}
                ],
                "optimal_strategy": {
                    "fire_commander": "send_maximum",
                    "medical_dispatch": "send_maximum",
                    "traffic_control": "evacuate"
                }
            },
            {
                "id": "S2",
                "description": "Flood threatening hospital + mass casualties",
                "crises": [
                    {"type": "flood", "severity": "critical", "location": "Zone B", "affected": 500},
                    {"type": "medical", "severity": "high", "location": "City Hospital", "affected": 200},
                    {"type": "infrastructure", "severity": "high", "location": "Zone B", "affected": 1000}
                ],
                "optimal_strategy": {
                    "fire_commander": "send_standard",
                    "medical_dispatch": "evacuate",
                    "traffic_control": "evacuate"
                }
            },
            {
                "id": "S3",
                "description": "Chemical plant explosion + mass casualties",
                "crises": [
                    {"type": "fire", "severity": "critical", "location": "Industrial Zone", "affected": 100},
                    {"type": "medical", "severity": "critical", "location": "Industrial Zone", "affected": 40},
                    {"type": "accident", "severity": "high", "location": "Zone C", "affected": 25}
                ],
                "optimal_strategy": {
                    "fire_commander": "evacuate",
                    "medical_dispatch": "send_maximum",
                    "traffic_control": "evacuate"
                }
            }
        ]

    def evaluate_coordination(
        self,
        agent_actions: Dict[str, str],
        scenario_id: str,
        step: int
    ) -> Tuple[float, str, Dict[str, float]]:
        """Evaluate coordination quality across all agents."""
        metrics = {}
        reward = 0.0
        feedback_parts = []

        scenario = next((s for s in self.scenarios if s["id"] == scenario_id), None)
        if not scenario:
            return 0.0, "Unknown scenario", {}

        optimal = scenario["optimal_strategy"]

        # Role correctness
        correct_count = sum(
            1 for role, action in agent_actions.items()
            if action == optimal.get(role, "")
        )
        role_score = correct_count / len(agent_actions) * 0.4
        reward += role_score
        metrics["role_score"] = role_score
        feedback_parts.append(f"Role accuracy: {correct_count}/{len(agent_actions)}")

        # Coordination check: agents should not conflict
        actions = list(agent_actions.values())
        if "evacuate" in actions and "send_minimum" in actions:
            conflict_penalty = -0.3
            reward += conflict_penalty
            metrics["conflict"] = conflict_penalty
            feedback_parts.append("⚠️ Conflicting actions detected")
        else:
            coord_bonus = 0.3
            reward += coord_bonus
            metrics["coordination"] = coord_bonus
            feedback_parts.append("✅ Agents synchronized")

        # Lives saved bonus (step 2+)
        if step >= 2 and correct_count >= 2:
            lives_bonus = 0.3
            reward += lives_bonus
            metrics["lives_saved"] = lives_bonus
            feedback_parts.append("✅ Lives saved")

        # Speed bonus
        if step <= 2 and correct_count == len(agent_actions):
            speed_bonus = 0.2
            reward += speed_bonus
            metrics["speed"] = speed_bonus
            feedback_parts.append("⚡ Fast resolution")

        final = min(1.0, max(-0.5, reward))
        return final, " | ".join(feedback_parts), metrics


# ─────────────────────────────────────────────
#  TASK 3 — EXTREME: Cascading Crisis
# ─────────────────────────────────────────────

class CascadingCrisisTask(Task):
    """
    Task 3 - EXTREME: Cascading Crisis Under Time Pressure

    A city-wide crisis that escalates if not handled correctly.
    Wrong decisions cause NEW crises to spawn (cascading effect).
    Time window is limited. Resources are critically scarce.

    Reward:
    - Cascade prevented:  +0.5
    - Lives saved:        +0.4 per person group
    - Cascade triggered:  -0.5 per new crisis
    - Timeout:            -1.0
    - Optimal path:       +1.0 total
    """

    def __init__(self):
        super().__init__(
            task_id=3,
            name="Cascading Crisis",
            description="Prevent city-wide cascading failures under extreme time pressure",
            difficulty="extreme",
            max_steps=4,
            num_agents=3
        )
        self.cascade_scenarios = [
            {
                "id": "CASCADE_1",
                "title": "Nuclear Plant Power Failure",
                "initial_crisis": {
                    "type": "infrastructure",
                    "severity": "critical",
                    "location": "Power Plant Zone",
                    "affected": 50000,
                    "time_window": 300  # seconds
                },
                "cascade_chain": [
                    {"trigger": "wrong_action", "spawns": "medical", "severity": "critical"},
                    {"trigger": "delay", "spawns": "flood", "severity": "high"},
                    {"trigger": "no_evacuate", "spawns": "fire", "severity": "critical"}
                ],
                "optimal_sequence": ["evacuate", "coordinate", "send_maximum", "coordinate"],
                "resources": {
                    "fire_truck": 3, "ambulance": 5,
                    "rescue_team": 2, "helicopter": 1
                }
            },
            {
                "id": "CASCADE_2",
                "title": "Mass Casualty Event at Stadium",
                "initial_crisis": {
                    "type": "medical",
                    "severity": "critical",
                    "location": "National Stadium",
                    "affected": 30000,
                    "time_window": 240
                },
                "cascade_chain": [
                    {"trigger": "send_minimum", "spawns": "accident", "severity": "high"},
                    {"trigger": "no_traffic_control", "spawns": "accident", "severity": "critical"},
                    {"trigger": "wrong_action", "spawns": "infrastructure", "severity": "high"}
                ],
                "optimal_sequence": ["send_maximum", "evacuate", "coordinate", "send_maximum"],
                "resources": {
                    "ambulance": 4, "police_car": 6,
                    "rescue_team": 3, "helicopter": 2
                }
            }
        ]

    def evaluate_step(
        self,
        action: str,
        step: int,
        cascading_risk: float,
        time_remaining: float,
        optimal_action: str
    ) -> Tuple[float, str, Dict[str, float], bool]:
        """
        Evaluate one step in cascading crisis.
        Returns: (reward, feedback, metrics, cascade_triggered)
        """
        metrics = {}
        reward = 0.0
        feedback_parts = []
        cascade_triggered = False

        if not action:
            return -0.5, "❌ No action taken", {"penalty": -0.5}, True

        action = action.lower().strip()

        # Correct action
        if action == optimal_action:
            base = 0.5
            reward += base
            metrics["correct"] = base
            feedback_parts.append(f"✅ Optimal: {action}")

            # Cascade prevention
            if cascading_risk > 0.6:
                cascade_bonus = 0.3
                reward += cascade_bonus
                metrics["cascade_prevented"] = cascade_bonus
                feedback_parts.append("🛡️ Cascade prevented")

        else:
            # Wrong action may trigger cascade
            penalty = -0.3
            reward += penalty
            metrics["wrong_action"] = penalty
            feedback_parts.append(f"❌ Suboptimal: {action}")

            if cascading_risk > 0.5:
                cascade_triggered = True
                cascade_penalty = -0.4
                reward += cascade_penalty
                metrics["cascade_triggered"] = cascade_penalty
                feedback_parts.append("💥 CASCADE TRIGGERED")

        # Time pressure bonus
        if time_remaining > 120 and action in ["evacuate", "send_maximum"]:
            time_bonus = 0.2
            reward += time_bonus
            metrics["time_bonus"] = time_bonus
            feedback_parts.append("⚡ Swift response")
        elif time_remaining < 30:
            time_penalty = -0.2
            reward += time_penalty
            metrics["time_pressure"] = time_penalty
            feedback_parts.append("⏰ Critical time pressure")

        final = min(1.0, max(-1.0, reward))
        return final, " | ".join(feedback_parts), metrics, cascade_triggered


# ─────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────

def create_task(task_id: int) -> Task:
    """Factory to create task by ID."""
    tasks = {
        0: CrisisClassificationTask,
        1: ResourceAllocationTask,
        2: MultiAgentCoordinationTask,
        3: CascadingCrisisTask
    }
    if task_id not in tasks:
        raise ValueError(f"Invalid task_id: {task_id}. Must be 0-3.")
    return tasks[task_id]()
