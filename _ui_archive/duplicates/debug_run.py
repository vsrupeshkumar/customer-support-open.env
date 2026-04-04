"""Quick diagnostic — runs each module in isolation."""
import sys, traceback
sys.path.insert(0, ".")

results = {}

# ── 1. Models ──────────────────────────────────────────────────────────────
try:
    from env.models import (
        Incident, IncidentType, SeverityLevel, IncidentStatus,
        ResourcePool, Observation, Action, EpisodeMetrics, AgentPerformance
    )
    pool = ResourcePool(fire_units=5, ambulances=6, traffic_units=3)
    assert pool.total() == 14
    assert pool.can_dispatch(2, 3, True)
    inc = Incident(
        incident_id="INC-001", incident_type=IncidentType.FIRE,
        severity=SeverityLevel.HIGH, location_id="Zone-A",
        people_affected=50, requires_fire_units=2, requires_ambulances=1,
        requires_traffic_control=True, time_to_critical=120.0,
    )
    assert inc.status == IncidentStatus.PENDING
    results["1_models"] = "PASS"
    print("✅ TEST 1 Models: PASS")
except Exception:
    results["1_models"] = "FAIL"
    print("❌ TEST 1 Models: FAIL")
    traceback.print_exc()

# ── 2. Reward ──────────────────────────────────────────────────────────────
try:
    from env.models import Action
    from env.reward import compute_reward
    a = Action(dispatch_fire_units=2, dispatch_ambulances=1,
               control_traffic=False, escalate_priority=True)
    r = compute_reward(a, inc, 10.0, {"fire_units": 8, "ambulances": 8, "traffic_units": 4})
    assert -1.0 <= r.value <= 1.0
    results["2_reward"] = "PASS"
    print(f"✅ TEST 2 Reward: PASS  value={r.value:+.3f}")
except Exception:
    results["2_reward"] = "FAIL"
    print("❌ TEST 2 Reward: FAIL")
    traceback.print_exc()

# ── 3. Tasks ───────────────────────────────────────────────────────────────
try:
    from env.tasks import create_task
    for tid in [1, 2, 3]:
        t = create_task(tid)
        incs, res = t.generate_episode(seed=42)
        assert len(incs) >= 1
    results["3_tasks"] = "PASS"
    print("✅ TEST 3 Tasks:  PASS")
except Exception:
    results["3_tasks"] = "FAIL"
    print("❌ TEST 3 Tasks:  FAIL")
    traceback.print_exc()

# ── 4. Environment ─────────────────────────────────────────────────────────
try:
    from env import CrisisManagementEnv
    from env.models import Action as A, Observation, EnvironmentState
    for tid in [1, 2, 3]:
        env = CrisisManagementEnv(task_id=tid, seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation)
        act = A(dispatch_fire_units=2, dispatch_ambulances=2,
                control_traffic=True, escalate_priority=False)
        obs2, rew, done, info = env.step(act)
        assert isinstance(rew, float) and -1.0 <= rew <= 1.0
        assert "reward_info" in info
        st = env.state()
        assert isinstance(st, EnvironmentState)
    results["4_env"] = "PASS"
    print("✅ TEST 4 Env:    PASS")
except Exception:
    results["4_env"] = "FAIL"
    print("❌ TEST 4 Env:    FAIL")
    traceback.print_exc()

# ── 5. Full Episode ────────────────────────────────────────────────────────
try:
    from env import CrisisManagementEnv
    from env.models import Action as A
    env = CrisisManagementEnv(task_id=2, seed=99)
    obs = env.reset()
    total_r = 0.0
    steps = 0
    while not env.is_done:
        act = A(
            dispatch_fire_units=2, dispatch_ambulances=2,
            control_traffic=True,
            escalate_priority=(obs.severity.value == "high")
        )
        obs, rew, done, info = env.step(act)
        total_r += rew
        steps += 1
        if done:
            break
    assert steps >= 1
    assert "episode_summary" in info
    score = info["episode_summary"]["final_score"]
    results["5_episode"] = "PASS"
    print(f"✅ TEST 5 Episode: PASS  steps={steps} score={score:.4f}")
except Exception:
    results["5_episode"] = "FAIL"
    print("❌ TEST 5 Episode: FAIL")
    traceback.print_exc()

# ── 6. Grader ──────────────────────────────────────────────────────────────
try:
    from env.grader import Grader
    from env.models import IncidentStatus
    grader = Grader()
    inc2 = inc.model_copy()
    inc2.status = IncidentStatus.RESOLVED
    inc2.time_dispatched = 20.0
    result = grader.grade_episode(1, [inc2], 20, 10, 60.0)
    assert 0.0 <= result.score <= 1.0
    grader.record_score(1, 0.9)
    grader.record_score(2, 0.7)
    grader.record_score(3, 0.5)
    summary = grader.get_summary()
    assert "Overall" in summary
    results["6_grader"] = "PASS"
    print(f"✅ TEST 6 Grader: PASS  score={result.score:.4f}")
except Exception:
    results["6_grader"] = "FAIL"
    print("❌ TEST 6 Grader: FAIL")
    traceback.print_exc()

# ── 7. Inference / Baseline Agent ─────────────────────────────────────────
try:
    from inference import BaselineAgent, run_episode
    agent = BaselineAgent()
    for tid in [1, 2, 3]:
        summary = run_episode(tid, agent, seed=42, verbose=False)
        sc = summary.get("final_score", 0.0)
        assert sc >= 0.0
        print(f"   Task {tid}: score={sc:.4f}")
    results["7_agent"] = "PASS"
    print("✅ TEST 7 Agent:  PASS")
except Exception:
    results["7_agent"] = "FAIL"
    print("❌ TEST 7 Agent:  FAIL")
    traceback.print_exc()

# ── Final Report ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for v in results.values() if v == "PASS")
total  = len(results)
print(f"  RESULT: {passed}/{total} tests passed")
for k, v in results.items():
    icon = "✅" if v == "PASS" else "❌"
    print(f"  {icon} {k}: {v}")
print("=" * 50)
sys.exit(0 if passed == total else 1)
