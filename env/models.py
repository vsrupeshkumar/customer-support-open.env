from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List

class FireLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CATASTROPHIC = "catastrophic"

class PatientLevel(str, Enum):
    NONE = "none"
    MODERATE = "moderate"
    CRITICAL = "critical"
    FATAL = "fatal"

class TrafficLevel(str, Enum):
    LOW = "low"
    HEAVY = "heavy"
    GRIDLOCK = "gridlock"

class WeatherCondition(str, Enum):
    CLEAR = "clear"
    STORM = "storm"
    HURRICANE = "hurricane"

class ResourcePool(BaseModel):
    fire_units: int = Field(default=0, ge=0)
    ambulances: int = Field(default=0, ge=0)
    police: int = Field(default=0, ge=0)

class ActiveDeployment(BaseModel):
    zone_id: str
    fire_units: int = 0
    ambulances: int = 0
    police: int = 0
    steps_remaining: int = 0

class ZoneState(BaseModel):
    fire: FireLevel = FireLevel.NONE
    patient: PatientLevel = PatientLevel.NONE
    traffic: TrafficLevel = TrafficLevel.LOW
    consecutive_failures: int = Field(default=0)

class Observation(BaseModel):
    """Spatial Multi-Node Triage State Vector"""
    weather: WeatherCondition
    zones: Dict[str, ZoneState]
    idle_resources: ResourcePool
    busy_resources: ResourcePool
    step: int = Field(default=0)
    max_steps: int = Field(default=10)

class ZoneDispatch(BaseModel):
    dispatch_fire: int = Field(default=0, ge=0)
    dispatch_ambulance: int = Field(default=0, ge=0)
    control_traffic: bool = Field(default=False)

class Action(BaseModel):
    """The RL Agent must dynamically route allocations across distinct spatial zones."""
    allocations: Dict[str, ZoneDispatch]

class EnvironmentState(BaseModel):
    step_count: int
    max_steps: int
    observation: Observation
    total_reward: float
    is_done: bool
    success: bool
    metrics: dict
