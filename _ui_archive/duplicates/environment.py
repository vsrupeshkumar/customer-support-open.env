"""
CrisisManagementEnv — Advanced Multi-Agent RL Environment

OpenEnv-compliant environment for AI crisis management.
Features:
- 4 tasks with increasing difficulty
- Multi-agent coordination (3 agents)
- Dynamic cascading crisis simulation
- Shaped reward signals
- Time pressure mechanics
- Resource constraints
"""

from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
import random
import time
import uuid

from models import (
    Observation, CrisisEvent, CrisisType, CrisisSeverity,
    AgentRole, CoordinationStatus, EnvironmentState,
    RewardBreakdown, Reward
)


# ─────────────────────────────────────────────
#  CITY ZONES & SCENARIO TEMPLATES
# ─────────────────────────────────────────────

CITY_ZONES = ["Zone A", "Zone B", "Zone C", "Industrial Zone",
              "Hospital District", "Downtown", "Suburbs", "Port Area"]

CRISIS_TEMPLATES = [
    {
        "crisis_type": "fire",
        "severity": "high",
        "location": "Industrial Zone",
        "affected": 150,
        "report": "Fire at chemical warehouse. Toxic smoke spreading."
    },
    {
        "crisis_type": "accident",
        "severity": "medium",
        "location": "Downtown",
        "affected": 20,
        "report": "Bus crash near central market. Multiple injuries reported."
    },
    {
        "crisis_type": "medical",
        "severity": "critical",
        "location": "Hospital District",
        "affected": 35,
        "report": "Mass poisoning at food festival. 35 people unconscious."
    },
    {
        "crisis_type": "flood",
        "severity": "critical",
        "location": "Zone B",
        "affected": 800,
        "report": "Storm surge flooding residential zone. People on rooftops."
    },
    {
        "crisis_type": "infrastructure",
        "severity": "high",
        "location": "Downtown",
        "affected": 5000,
        "report": "Major power grid failure. Hospital backup systems at risk."
    },
    {
        "crisis_type": "fire",
        "severity": "critical",
        "location": "Zone A",
        "affected": 300,
        "report": "City mall fire. Hundreds trapped inside."
    },
    {
        "crisis_type": "flood",
        "severity": "high",
        "location": "Suburbs",
        "affected": 400,
        "report": "Flash flood after dam overflow. Evacuation needed."
    },
    {
        "crisis_type": "accident",
        "severity": "critical",
        "location": "Port Area",
        "affected": 60,
        "report": "Ship collision at port. Crew members in water."
    },
    {
        "crisis_type": "medical",
        "severity": "high",
        "location": "Zone C",
        "affected": 50,
        "report": "Gas leak causing mass respiratory issues in apartment block."
    },
    {
        "crisis_type": "infrastructure",
        "severity": "critical",
        "location": "Zone A",
        "affected": 10000,
        "report": "Bridge collapse on main highway. Multiple vehicles fallen."
    }
]

AVAILABLE_RESOURCES_POOL = {
    "fire_truck": 8,
    "ambulance": 10,
    "police_car": 12,
    "rescue_team": 6,
    "helicopter": 3
}


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────

@dataclass
class CrisisManagementEnv:
    """
    Multi-Agent Reinforcement Learning Environment for Crisis Management.

    Implements OpenEnv spec:
    - reset()  → Observation
    - step()   → (Observation, float, bool, dict)
    - state()  → EnvironmentState

    Tasks:
    - 0: Crisis Classification (Easy)
    - 1: Resource Allocation   (Medium)
    - 2: Multi-Agent Coord     (Hard)
    - 3: Cascading Crisis      (Extreme)
    """

    task_id: int = 0

    # Episode state
    step_count: int = 0
    max_steps: int = 1
    is_resolved: bool = False
    episode_start_time: float = 0.0
    time_elapsed: float = 0.0

    # Crisis state
    active_crises: List[CrisisEvent] = field(default_factory=list)
    resolved_crises: List[CrisisEvent] = field(default_factory=list)
    current_scenario: Dict = field(default_factory=dict)

    # Resources
    available_resources: Dict[str, int] = field(default_factory=dict)
    resources_used: Dict[str, int] = field(default_factory=dict)

    # Lives
    lives_at_risk: int = 0
    lives_saved: int = 0
    lives_lost: int = 0

    # Multi-agent state
    agent_roles: List[str] = field(default_factory=list)
    current_agent_role: str = ""
    agent_actions_this_step: Dict[str, str] = field(default_factory=dict)
    coordination_score: float = 0.0

    # Task 3 state
    cascading_risk: float = 0.0
    cascade_events: List[Dict] = field(default_factory=list)
    time_remaining: float = 300.0
    current_optimal_action: str = ""

    # Reward tracking
    total_reward: float = 0.0
    mistakes_count: int = 0
    step_history: List[Dict] = field(default_factory=list)
    last_feedback: str = ""
    last_is_correct: bool = False
    last_quality_metrics: Dict = field(default_factory=dict)

    # ──────────────────────────────── reset ────

    def reset(self) -> Observation:
        """Reset environment for a new episode. Returns initial observation."""
        self.step_count = 0
        self.is_resolved = False
        self.lives_saved = 0
        self.lives_lost = 0
        self.total_reward = 0.0
        self.mistakes_count = 0
        self.step_history = []
        self.agent_actions_this_step = {}
        self.resolved_crises = []
        self.cascade_events = []
        self.last_feedback = ""
        self.last_is_correct = False
        self.last_quality_metrics = {}
        self.episode_start_time = time.time()

        if self.task_id == 0:
            self.max_steps = 1
            self._init_task0()
        elif self.task_id == 1:
            self.max_steps = 2
            self._init_task1()
        elif self.task_id == 2:
            self.max_steps = 5
            self._init_task2()
        else:
            self.max_steps = 4
            self._init_task3()

        return self._build_observation()

    def _init_task0(self):
        """Single crisis for classification."""
        template = random.choice(CRISIS_TEMPLATES)
        self.current_scenario = template
        crisis = CrisisEvent(
            event_id=f"E{uuid.uuid4().hex[:6].upper()}",
            crisis_type=CrisisType(template["crisis_type"]),
            severity=CrisisSeverity(template["severity"]),
            location=template["location"],
            affected_people=template["affected"]
        )
        self.active_crises = [crisis]
        self.lives_at_risk = template["affected"]
        self.available_resources = {}
        self.agent_roles = ["commander"]

    def _init_task1(self):
        """Crisis + resource constraints."""
        from tasks import ResourceAllocationTask
        task = ResourceAllocationTask()
        scenario = random.choice(task.scenarios)
        self.current_scenario = scenario
        crisis = CrisisEvent(
            event_id=f"E{uuid.uuid4().hex[:6].upper()}",
            crisis_type=CrisisType(scenario["crisis_type"]),
            severity=CrisisSeverity(scenario["severity"]),
            location="Zone A",
            affected_people=random.randint(10, 200)
        )
        self.active_crises = [crisis]
        self.lives_at_risk = crisis.affected_people
        self.available_resources = dict(scenario["available"])
        self.agent_roles = ["commander"]

    def _init_task2(self):
        """3 simultaneous crises, 3 agents."""
        from tasks import MultiAgentCoordinationTask
        task = MultiAgentCoordinationTask()
        scenario = random.choice(task.scenarios)
        self.current_scenario = scenario

        self.active_crises = []
        total_affected = 0
        for c in scenario["crises"]:
            crisis = CrisisEvent(
                event_id=f"E{uuid.uuid4().hex[:6].upper()}",
                crisis_type=CrisisType(c["type"]),
                severity=CrisisSeverity(c["severity"]),
                location=c["location"],
                affected_people=c["affected"]
            )
            self.active_crises.append(crisis)
            total_affected += c["affected"]

        self.lives_at_risk = total_affected
        self.available_resources = dict(AVAILABLE_RESOURCES_POOL)
        self.agent_roles = ["fire_commander", "medical_dispatch", "traffic_control"]
        self.current_agent_role = random.choice(self.agent_roles)
        self.coordination_score = 0.5
        self.cascading_risk = 0.3

    def _init_task3(self):
        """Cascading crisis, extreme difficulty."""
        from tasks import CascadingCrisisTask
        task = CascadingCrisisTask()
        scenario = random.choice(task.cascade_scenarios)
        self.current_scenario = scenario

        ic = scenario["initial_crisis"]
        crisis = CrisisEvent(
            event_id=f"E{uuid.uuid4().hex[:6].upper()}",
            crisis_type=CrisisType(ic["type"]),
            severity=CrisisSeverity(ic["severity"]),
            location=ic["location"],
            affected_people=ic["affected"]
        )
        self.active_crises = [crisis]
        self.lives_at_risk = ic["affected"]
        self.available_resources = dict(scenario["resources"])
        self.agent_roles = ["fire_commander", "medical_dispatch", "traffic_control"]
        self.current_agent_role = "fire_commander"
        self.cascading_risk = 0.6
        self.time_remaining = float(ic["time_window"])
        self.current_optimal_action = scenario["optimal_sequence"][0]

    # ──────────────────────────────── step ────

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one step. Returns (obs, reward, done, info)."""
        self.step_count += 1
        self.time_elapsed = time.time() - self.episode_start_time
        done = False
        reward_value = 0.0

        if self.task_id == 0:
            reward_value, done = self._step_task0(action)
        elif self.task_id == 1:
            reward_value, done = self._step_task1(action)
        elif self.task_id == 2:
            reward_value, done = self._step_task2(action)
        else:
            reward_value, done = self._step_task3(action)

        self.total_reward += reward_value
        if self.step_count >= self.max_steps:
            done = True

        obs = self._build_observation()
        self.step_history.append({
            "step": self.step_count,
            "action": action,
            "reward": round(reward_value, 3),
            "feedback": self.last_feedback
        })

        info = {
            "step": self.step_count,
            "is_resolved": self.is_resolved,
            "lives_saved": self.lives_saved,
            "lives_lost": self.lives_lost,
            "total_reward": round(self.total_reward, 3),
            "feedback": self.last_feedback,
            "cascading_risk": self.cascading_risk,
            "reward_info": {
                "is_correct": self.last_is_correct,
                "quality_metrics": self.last_quality_metrics,
                "feedback": self.last_feedback
            }
        }

        return obs, reward_value, done, info

    def _step_task0(self, action: str) -> Tuple[float, bool]:
        from tasks import CrisisClassificationTask
        task = CrisisClassificationTask()
        expected = self.current_scenario["crisis_type"]
        score, feedback = task.evaluate(action, expected)
        self.last_is_correct = (score == 1.0)
        self.last_feedback = feedback
        self.last_quality_metrics = {"classification": score}
        if self.last_is_correct:
            self.lives_saved = self.lives_at_risk // 2
            self.is_resolved = True
        else:
            self.mistakes_count += 1
        return score, True

    def _step_task1(self, action: str) -> Tuple[float, bool]:
        from tasks import ResourceAllocationTask
        task = ResourceAllocationTask()
        severity = self.current_scenario["severity"]
        crisis_type = self.current_scenario["crisis_type"]
        score, feedback = task.evaluate(action, severity, crisis_type)
        self.last_is_correct = (score >= 0.9)
        self.last_feedback = feedback
        self.last_quality_metrics = {"allocation": score}

        if score >= 0.9:
            self.lives_saved = int(self.lives_at_risk * 0.9)
            self.is_resolved = True
        elif score >= 0.5:
            self.lives_saved = int(self.lives_at_risk * 0.6)
        else:
            self.lives_lost = int(self.lives_at_risk * 0.3)
            self.mistakes_count += 1

        return score, True

    def _step_task2(self, action: str) -> Tuple[float, bool]:
        from tasks import MultiAgentCoordinationTask
        task = MultiAgentCoordinationTask()

        self.agent_actions_this_step[self.current_agent_role] = action

        # Simulate other agents acting
        scenario = self.current_scenario
        optimal = scenario.get("optimal_strategy", {})

        for role in self.agent_roles:
            if role not in self.agent_actions_this_step:
                # Other agents follow optimal with 70% probability
                if random.random() < 0.7:
                    self.agent_actions_this_step[role] = optimal.get(role, "send_standard")
                else:
                    self.agent_actions_this_step[role] = random.choice(
                        ["send_minimum", "send_standard", "send_maximum", "evacuate"]
                    )

        score, feedback, metrics = task.evaluate_coordination(
            self.agent_actions_this_step,
            scenario.get("id", "S1"),
            self.step_count
        )

        self.last_is_correct = (score >= 0.7)
        self.last_feedback = feedback
        self.last_quality_metrics = metrics
        self.coordination_score = min(1.0, self.coordination_score + (0.1 if score > 0.5 else -0.05))

        if score >= 0.7:
            self.lives_saved += int(self.lives_at_risk * 0.15)

        if self.step_count >= 3 and score >= 0.6:
            self.is_resolved = True

        # Reset for next step
        self.agent_actions_this_step = {}
        self.current_agent_role = random.choice(self.agent_roles)

        done = self.is_resolved or self.step_count >= self.max_steps
        return score, done

    def _step_task3(self, action: str) -> Tuple[float, bool]:
        from tasks import CascadingCrisisTask
        task = CascadingCrisisTask()

        # Update time
        self.time_remaining = max(0.0, self.time_remaining - 60.0)

        # Get optimal action for this step
        optimal_seq = self.current_scenario.get("optimal_sequence", [])
        optimal = optimal_seq[self.step_count - 1] if self.step_count <= len(optimal_seq) else "coordinate"

        score, feedback, metrics, cascade_triggered = task.evaluate_step(
            action,
            self.step_count,
            self.cascading_risk,
            self.time_remaining,
            optimal
        )

        self.last_is_correct = not cascade_triggered and score >= 0.4
        self.last_feedback = feedback
        self.last_quality_metrics = metrics

        if cascade_triggered:
            # Spawn new crisis
            new_crisis_template = random.choice(CRISIS_TEMPLATES)
            new_crisis = CrisisEvent(
                event_id=f"CASCADE_{uuid.uuid4().hex[:4].upper()}",
                crisis_type=CrisisType(new_crisis_template["crisis_type"]),
                severity=CrisisSeverity("high"),
                location=random.choice(CITY_ZONES),
                affected_people=random.randint(50, 300)
            )
            self.active_crises.append(new_crisis)
            self.lives_at_risk += new_crisis.affected_people
            self.cascading_risk = min(1.0, self.cascading_risk + 0.2)
            self.cascade_events.append({"step": self.step_count, "crisis": new_crisis_template["crisis_type"]})
        else:
            self.cascading_risk = max(0.0, self.cascading_risk - 0.15)
            self.lives_saved += int(self.lives_at_risk * 0.1)

        if self.time_remaining <= 0:
            score -= 0.5
            self.last_feedback += " | ⏰ TIME EXPIRED"
            self.lives_lost += int(self.lives_at_risk * 0.4)

        if self.step_count >= self.max_steps and not cascade_triggered:
            self.is_resolved = True

        done = self.time_remaining <= 0 or self.is_resolved or self.step_count >= self.max_steps
        return score, done

    # ──────────────────────────────── observation ────

    def _build_observation(self) -> Observation:
        """Build typed observation for current task."""
        history = self.step_history[-3:] if len(self.step_history) > 3 else self.step_history

        if self.task_id == 0:
            primary = self.active_crises[0] if self.active_crises else None
            return Observation(
                task_id=self.task_id,
                step=self.step_count,
                max_steps=self.max_steps,
                crisis_events=self.active_crises,
                primary_crisis_type=primary.crisis_type if primary else None,
                lives_at_risk=self.lives_at_risk,
                step_history=history
            )

        elif self.task_id == 1:
            primary = self.active_crises[0] if self.active_crises else None
            return Observation(
                task_id=self.task_id,
                step=self.step_count,
                max_steps=self.max_steps,
                crisis_events=self.active_crises,
                primary_crisis_type=primary.crisis_type if primary else None,
                primary_severity=primary.severity if primary else None,
                available_resources=self.available_resources,
                time_pressure=0.5,
                lives_at_risk=self.lives_at_risk,
                step_history=history
            )

        elif self.task_id == 2:
            return Observation(
                task_id=self.task_id,
                step=self.step_count,
                max_steps=self.max_steps,
                crisis_events=self.active_crises,
                available_resources=self.available_resources,
                time_pressure=min(1.0, len(self.active_crises) * 0.3),
                active_agents=self.agent_roles,
                agent_role=AgentRole(self.current_agent_role) if self.current_agent_role else None,
                coordination_status=CoordinationStatus.ACTIVE,
                other_agents_actions=self.agent_actions_this_step,
                lives_at_risk=self.lives_at_risk,
                step_history=history
            )

        else:  # Task 3
            return Observation(
                task_id=self.task_id,
                step=self.step_count,
                max_steps=self.max_steps,
                crisis_events=self.active_crises,
                available_resources=self.available_resources,
                time_pressure=1.0 - (self.time_remaining / 300.0),
                active_agents=self.agent_roles,
                agent_role=AgentRole(self.current_agent_role) if self.current_agent_role else None,
                coordination_status=CoordinationStatus.ACTIVE,
                cascading_risk=self.cascading_risk,
                time_remaining=self.time_remaining,
                lives_at_risk=self.lives_at_risk,
                step_history=history
            )

    # ──────────────────────────────── state ────

    def state(self) -> EnvironmentState:
        """Return full internal state as EnvironmentState."""
        self.time_elapsed = time.time() - self.episode_start_time
        return EnvironmentState(
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            active_crises=self.active_crises,
            resolved_crises=self.resolved_crises,
            available_resources=self.available_resources,
            total_lives_at_risk=self.lives_at_risk,
            lives_saved=self.lives_saved,
            lives_lost=self.lives_lost,
            time_elapsed=self.time_elapsed,
            cascading_risk=self.cascading_risk,
            coordination_score=self.coordination_score,
            total_reward=self.total_reward,
            agent_roles=self.agent_roles,
            is_resolved=self.is_resolved,
            episode_success=self.is_resolved and self.lives_lost == 0
        )
