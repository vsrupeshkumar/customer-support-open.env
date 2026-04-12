import pytest

from env.reward import calculate_nlp_bonus
from env.models import Observation, WeatherCondition, ZoneState, FireLevel, PatientLevel, TrafficLevel, ResourcePool

@pytest.fixture
def base_obs():
    zones = {
        "Downtown": ZoneState(fire=FireLevel.CATASTROPHIC, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW),
        "Harbor": ZoneState(fire=FireLevel.NONE, patient=PatientLevel.NONE, traffic=TrafficLevel.LOW),
    }
    return Observation(
        weather=WeatherCondition.CLEAR,
        zones=zones,
        idle_resources=ResourcePool(fire_units=10, ambulances=10, police=10),
        busy_resources=ResourcePool()
    )


def test_perfect_broadcast(base_obs):
    msg = "WARNING: Downtown has a severe fire. Evacuate immediately."
    score = calculate_nlp_bonus(msg, base_obs)
    # Zone match (0.4) + hazard match (0.3) + action match (0.3)
    # Actually checking the implementation in calculate_nlp_bonus to see exact reward
    assert score > 0.8, f"Score should be close to 1.0, got {score}"

def test_hallucination_penalty(base_obs):
    # Mention Harbor, which has no fire
    msg = "Harbor fire"
    score = calculate_nlp_bonus(msg, base_obs)
    # Harbor mentioned = hallucination
    # Downtown not mentioned
    assert score < 0.0, f"Score should be negative due to hallucination penalty, got {score}"

def test_bloat_penalty(base_obs):
    perfect_msg = "WARNING: Downtown has a severe fire. Evacuate immediately."
    perfect_score = calculate_nlp_bonus(perfect_msg, base_obs)
    
    # 80 words vs 50 threshold: 30 * 0.01 = 0.30 penalty
    bloated_msg = perfect_msg + " " + "blah " * 60
    bloat_score = calculate_nlp_bonus(bloated_msg, base_obs)
    
    assert bloat_score < perfect_score, "Bloated message should be penalized."

def test_empty_message_returns_zero(base_obs):
    assert calculate_nlp_bonus("", base_obs) == 0.0, "Empty message must yield identically 0.0"
    assert calculate_nlp_bonus(None, base_obs) == 0.0, "None message must yield identically 0.0"

def test_no_zero_floor_clamp(base_obs):
    msg = "Harbor Harbor Harbor Harbor" 
    # Hallucinations only, no clamping to 0!
    score = calculate_nlp_bonus(msg, base_obs)
    assert score < 0.0, "Negative values must NOT be clamped to 0.0 (Directive 3 compliance)."
