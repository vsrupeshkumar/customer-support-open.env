"""
Pydantic models for AI Crisis Management OpenEnv.

Typed models for all I/O in the multi-agent crisis management
reinforcement learning environment. Fully OpenEnv spec compliant.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────

class CrisisType(str, Enum):
    FIRE         = "fire"
    ACCIDENT     = "accident"
    MEDICAL      = "medical"
    FLOOD        = "flood"
    INFRASTRUCTURE = "infrastructure"


class CrisisSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    FIRE_COMMANDER   = "fire_commander"
    MEDICAL_DISPATCH = "medical_dispatch"
    TRAFFIC_CONTROL  = "traffic_control"


class ResourceType(str, Enum):
    FIRE_TRUCK  = "fire_truck"
    AMBULANCE   = "ambulance"
    POLICE_CAR  = "police_car"
    RESCUE_TEAM = "rescue_team"
    HELICOPTER  = "helicopter"


class ActionType(str, Enum):
    # Task 0 — Classification
    FIRE           = "fire"
    ACCIDENT       = "accident"
    MEDICAL        = "medical"
    FLOOD          = "flood"
    INFRASTRUCTURE = "infrastructure"

    # Task 1 — Resource Allocation
    SEND_MINIMUM  = "send_minimum"
    SEND_STANDARD = "send_standard"
    SEND_MAXIMUM  = "send_maximum"
    EVACUATE      = "evacuate"

    # Task 2/3 — Coordination
    COORDINATE    = "coordinate"
    ESCALATE      = "escalate"
    HOLD          = "hold"
    REASSIGN      = "reassign"


class CoordinationStatus(str, Enum):
    PENDING     = "pending"
    ACTIVE      = "active"
    RESOLVED    = "resolved"
    FAILED      = "failed"


# ─────────────────────────────────────────────
#  CRISIS EVENT
# ─────────────────────────────────────────────

class CrisisEvent(BaseModel):
    """A single crisis event in the city."""

    event_id: str = Field(..., description="Unique event identifier")
    crisis_type: CrisisType = Field(..., description="Type of crisis")
    severity: CrisisSeverity = Field(..., description="Severity level")
    location: str = Field(..., description="Crisis location (zone/district)")
    affected_people: int = Field(..., ge=0, description="Number of people affected")
    time_elapsed: float = Field(default=0.0, ge=0.0, description="Minutes since crisis started")
    is_active: bool = Field(default=True, description="Whether crisis is still active")
    resources_assigned: Dict[str, int] = Field(
        default_factory=dict,
        description="Resources currently assigned {resource_type: count}"
    )

    class Config:
        use_enum_values = False


# ─────────────────────────────────────────────
#  OBSERVATION
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """
    Typed observation returned by the environment.

    Different fields exposed based on task difficulty:
    - Task 0: crisis_events only (classify)
    - Task 1: events + available_resources (allocate)
    - Task 2: all fields (multi-agent coordinate)
    - Task 3: all fields + cascading_risk (extreme)
    """

    task_id: int = Field(..., description="Current task (0-3)")
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=5, description="Max steps allowed")

    # Core crisis info (all tasks)
    crisis_events: List[CrisisEvent] = Field(
        default_factory=list,
        description="Active crisis events in the city"
    )
    primary_crisis_type: Optional[CrisisType] = Field(
        default=None,
        description="Main crisis type (Task 0)"
    )
    primary_severity: Optional[CrisisSeverity] = Field(
        default=None,
        description="Primary crisis severity (Task 1+)"
    )

    # Task 1+ fields
    available_resources: Optional[Dict[str, int]] = Field(
        default=None,
        description="Available resources by type"
    )
    time_pressure: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Urgency level 0.0 (calm) to 1.0 (extreme urgency)"
    )

    # Task 2+ fields
    active_agents: Optional[List[str]] = Field(
        default=None,
        description="Roles of active agents in this episode"
    )
    agent_role: Optional[AgentRole] = Field(
        default=None,
        description="The current agent's role"
    )
    coordination_status: Optional[CoordinationStatus] = Field(
        default=None,
        description="Current coordination state between agents"
    )
    other_agents_actions: Optional[Dict[str, str]] = Field(
        default=None,
        description="Actions taken by other agents this step"
    )

    # Task 3 fields
    cascading_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Risk of cascading failures (Task 3 only)"
    )
    time_remaining: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Seconds remaining before critical threshold"
    )

    # Always present
    lives_at_risk: int = Field(default=0, ge=0, description="People currently in danger")
    step_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous steps summary"
    )

    class Config:
        use_enum_values = False


# ─────────────────────────────────────────────
#  ACTION
# ─────────────────────────────────────────────

class Action(BaseModel):
    """Typed action submitted to the environment."""

    action: str = Field(
        ...,
        min_length=1,
        description="Action string. Format depends on task."
    )
    resources_allocated: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional resource allocation map"
    )
    target_event_id: Optional[str] = Field(
        default=None,
        description="Which crisis event this action targets"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional agent reasoning (for LLM grading)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"action": "fire"},                                # Task 0
                {"action": "send_maximum"},                        # Task 1
                {"action": "coordinate", "target_event_id": "E1"} # Task 2
            ]
        }


# ─────────────────────────────────────────────
#  REWARD
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed reward component breakdown."""

    lives_saved: float = Field(default=0.0, description="+10 per life saved")
    response_speed: float = Field(default=0.0, description="Speed bonus/penalty")
    resource_efficiency: float = Field(default=0.0, description="Efficiency score")
    coordination_bonus: float = Field(default=0.0, description="Multi-agent sync bonus")
    cascade_prevention: float = Field(default=0.0, description="Prevented cascades")
    penalties: float = Field(default=0.0, description="All penalties combined")
    total: float = Field(default=0.0, description="Final reward value")


class Reward(BaseModel):
    """Full typed reward for OpenEnv compliance."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Normalized reward [-1, 1]")
    raw_score: float = Field(..., description="Raw score before normalization")
    breakdown: RewardBreakdown = Field(..., description="Component breakdown")
    feedback: str = Field(default="", description="Human-readable feedback")
    is_correct: bool = Field(default=False, description="Optimal action taken?")


# ─────────────────────────────────────────────
#  ENVIRONMENT STATE
# ─────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Full internal state — returned by state() method."""

    task_id: int
    step_count: int
    max_steps: int
    active_crises: List[CrisisEvent] = Field(default_factory=list)
    resolved_crises: List[CrisisEvent] = Field(default_factory=list)
    available_resources: Dict[str, int] = Field(default_factory=dict)
    total_lives_at_risk: int = 0
    lives_saved: int = 0
    lives_lost: int = 0
    time_elapsed: float = 0.0
    cascading_risk: float = 0.0
    coordination_score: float = 0.0
    total_reward: float = 0.0
    agent_roles: List[str] = Field(default_factory=list)
    is_resolved: bool = False
    episode_success: bool = False


# ─────────────────────────────────────────────
#  METRICS & PERFORMANCE
# ─────────────────────────────────────────────

class EpisodeMetrics(BaseModel):
    """Metrics from a single completed episode."""

    task_id: int
    total_reward: float
    steps_taken: int
    lives_saved: int
    lives_lost: int
    crises_resolved: int
    crises_failed: int
    avg_response_time: float = 0.0
    coordination_score: float = 0.0
    cascade_prevented: bool = False
    success: bool = False


class AgentPerformance(BaseModel):
    """Aggregate agent performance across episodes."""

    agent_name: str
    total_episodes: int = 0
    task_scores: Dict[int, float] = Field(default_factory=dict)
    overall_average: float = 0.0
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_lives_saved: float = 0.0
    avg_response_time: float = 0.0
    coordination_efficiency: float = 0.0
