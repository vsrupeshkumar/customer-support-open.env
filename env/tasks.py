from env.models import Observation, ResourcePool, FireLevel, PatientLevel, TrafficLevel, WeatherCondition, ZoneState

class Task:
    task_id: int
    name: str
    def generate_initial_observation(self, seed: int = None) -> Observation:
        pass

class EasyTask(Task):
    task_id = 1
    name = "Single-Zone Emergency"
    def generate_initial_observation(self, seed=None) -> Observation:
        zones = {
            "Downtown": ZoneState(fire=FireLevel.MEDIUM, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW),
            "Suburbs": ZoneState(),
            "Industrial": ZoneState()
        }
        return Observation(
            weather=WeatherCondition.CLEAR,
            zones=zones,
            idle_resources=ResourcePool(fire_units=5, ambulances=5, police=3),
            busy_resources=ResourcePool(fire_units=0, ambulances=0, police=0),
            step=0, max_steps=8
        )

class MediumTask(Task):
    task_id = 2
    name = "Multi-Zone Weather Chaos"
    def generate_initial_observation(self, seed=None) -> Observation:
        zones = {
            "Downtown": ZoneState(fire=FireLevel.NONE, patient=PatientLevel.MODERATE, traffic=TrafficLevel.HEAVY),
            "Suburbs": ZoneState(fire=FireLevel.HIGH, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW),
            "Industrial": ZoneState()
        }
        return Observation(
            weather=WeatherCondition.STORM,
            zones=zones,
            idle_resources=ResourcePool(fire_units=5, ambulances=3, police=2),
            busy_resources=ResourcePool(),
            step=0, max_steps=10
        )

class HardTask(Task):
    task_id = 3
    name = "City-Wide Meta Triage"
    def generate_initial_observation(self, seed=None) -> Observation:
        zones = {
            "Downtown": ZoneState(fire=FireLevel.HIGH, patient=PatientLevel.NONE, traffic=TrafficLevel.GRIDLOCK),
            "Suburbs": ZoneState(fire=FireLevel.NONE, patient=PatientLevel.CRITICAL, traffic=TrafficLevel.GRIDLOCK),
            "Industrial": ZoneState(fire=FireLevel.CATASTROPHIC, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW)
        }
        return Observation(
            weather=WeatherCondition.HURRICANE,
            zones=zones,
            idle_resources=ResourcePool(fire_units=8, ambulances=4, police=2), # Not enough to save all 3 instantly
            busy_resources=ResourcePool(),
            step=0, max_steps=12
        )

def create_task(task_id: int) -> Task:
    if task_id == 1: return EasyTask()
    if task_id == 2: return MediumTask()
    if task_id == 3: return HardTask()
    raise ValueError(f"Invalid task ID: {task_id}")
