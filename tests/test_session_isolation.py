import pytest
from fastapi.testclient import TestClient

from server.app import app, _sessions

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_sessions():
    """Ensure sessions are cleared before and after each test."""
    _sessions.clear()
    yield
    _sessions.clear()

def test_concurrent_sessions_dont_bleed():
    # Create two isolated sessions
    res_a = client.post("/reset", json={"task_id": 1, "seed": 42, "session_id": "session_A"})
    res_b = client.post("/reset", json={"task_id": 2, "seed": 99, "session_id": "session_B"})
    
    assert res_a.status_code == 200
    assert res_b.status_code == 200
    
    obs_a = res_a.json()
    obs_b = res_b.json()
    
    assert obs_a["session_id"] == "session_A"
    assert obs_b["session_id"] == "session_B"
    assert obs_a != obs_b, "Different tasks with different seeds must not have the exact same state"
    
    # Send empty actions
    action_a = {"session_id": "session_A", "allocations": {}}
    action_b = {"session_id": "session_B", "allocations": {}}
    
    step_a = client.post("/step", json=action_a)
    step_b = client.post("/step", json=action_b)
    
    assert step_a.status_code == 200
    assert step_b.status_code == 200
    
    # Just asserting isolated state runs correctly under correct IDs
    assert step_a.json()["observation"] != step_b.json()["observation"]

def test_session_capacity_limit():
    from server.app import MAX_SESSIONS
    
    # Create MAX_SESSIONS valid sessions
    for i in range(MAX_SESSIONS):
        resp = client.post("/reset", json={"task_id": 1, "session_id": f"sess_{i}"})
        assert resp.status_code == 200, f"Session {i} creation failed."
        
    # Exceed capacity
    resp_exceed = client.post("/reset", json={"task_id": 1, "session_id": "excess_session"})
    assert resp_exceed.status_code == 429, "Exceeding session capacity must return HTTP 429"
    assert "Session capacity reached" in resp_exceed.json()["detail"]
