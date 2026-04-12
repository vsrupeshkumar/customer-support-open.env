#!/usr/bin/env python3
"""
benchmark.py
============
Baseline Variance Proof — Phase 3 Real-world Utility Validation

Runs three agent types (Random, Heuristic, LLM) against all three tasks
across multiple seeds to demonstrate that the environment produces
meaningful, discriminative score variance.

Agent Architecture
------------------
1. **RandomAgent**: Uniform random dispatch within idle resource bounds.
   Expected score range: [0.05, 0.25] — should NOT be zero due to
   partial-progress gradient in the reward function.

2. **HeuristicAgent**: Priority-based dispatch using the exact mathematical
   requirement functions from env/reward.py (_get_required_fire,
   _get_required_ambulance). Allocates the minimum required resources to
   zones in descending severity order.
   Expected score range: [0.40, 0.75] — demonstrates competent play
   without LLM reasoning.

3. **LLMAgent** (optional): The full inference.py LLM agent.
   Expected score range: [0.50, 0.90] — requires API credentials.

Mathematical Foundation
-----------------------
For each (agent, task, seed) triplet, the benchmark records:
  - Episode score:  grade_episode(...)  ∈ [0.0, 1.0]
  - Efficiency:     _compute_efficiency(total_reward, total_incidents)  ∈ [0.0, 1.0]
  - Total reward:   cumulative step-reward sum

Statistical output:  mean ± std over N_SEEDS for each (agent, task) pair.

Usage
-----
    # Run Random + Heuristic only (no API key needed):
    python benchmark.py

    # Include LLM agent (requires HF_TOKEN):
    python benchmark.py --include-llm

    # Custom seed count:
    python benchmark.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_URL = "http://localhost:7860"
N_SEEDS = 10
TASKS = [1, 2, 3]
TASK_NAMES = {1: "Single-Zone Emergency (Easy)", 2: "Multi-Zone Weather Chaos (Med)", 3: "City-Wide Meta Triage (Hard)"}

# ---------------------------------------------------------------------------
# Data contracts (mirror env/models.py enums as strings for API JSON layer)
# ---------------------------------------------------------------------------

# Ordinal severity ranks — mirrors env/reward.py _FIRE_RANK / _PATIENT_RANK
FIRE_RANK   = {"none": 0, "low": 1, "medium": 2, "high": 3, "catastrophic": 4}
PATIENT_RANK = {"none": 0, "moderate": 1, "critical": 2, "fatal": 3}

# Minimum dispatch requirements — mirrors env/reward.py _get_required_fire / _get_required_ambulance
FIRE_REQ = {"catastrophic": 5, "high": 3, "medium": 2, "low": 1, "none": 0}
AMB_REQ  = {"critical": 3, "moderate": 1, "fatal": 0, "none": 0}

# Weather friction modifiers — mirrors RewardConstants
WEATHER_FIRE_FRICTION = {"hurricane": 2, "storm": 1, "clear": 0}


def get_required_fire(fire_level: str, weather: str) -> int:
    """Compute minimum fire units needed, including weather friction.
    
    Mathematical mapping:
        req = FIRE_REQ[fire_level]
        if req > 0: req += WEATHER_FIRE_FRICTION[weather]
    """
    base = FIRE_REQ.get(fire_level, 0)
    if base > 0:
        base += WEATHER_FIRE_FRICTION.get(weather, 0)
    return base


def get_required_ambulance(patient_level: str) -> int:
    """Compute minimum ambulances needed.
    
    Mathematical mapping:
        CRITICAL → 3, MODERATE → 1, FATAL/NONE → 0
    """
    return AMB_REQ.get(patient_level, 0)


# ---------------------------------------------------------------------------
# Episode result container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    task_id: int
    seed: int
    agent_name: str
    score: float
    efficiency: float
    total_reward: float
    steps: int


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class BaseAgent:
    """Abstract base for benchmark agents."""
    name: str = "base"

    def get_action(self, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Return an action dict compatible with POST /step."""
        raise NotImplementedError

    def reset(self) -> None:
        """Called between episodes."""
        pass


# ---------------------------------------------------------------------------
# Agent 1: Random Agent
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    """Uniform random dispatch within idle resource bounds.
    
    For each zone with an active hazard, dispatches a random number of
    fire units ∈ [0, min(2, idle_fire)] and ambulances ∈ [0, min(2, idle_amb)].
    Randomly enables traffic control with p=0.3.
    
    This agent is deliberately NON-trivial: it doesn't dispatch zero everywhere
    (which would trigger the Anti-Exploit Guard), nor does it flood all
    resources into one zone (which would cause inventory breach). It
    samples from a bounded uniform distribution, producing scores in the
    range [0.05, 0.25] that are meaningfully above zero.
    """
    name = "Random"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def get_action(self, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
        zones = obs.get("zones", {})
        idle = obs.get("idle_resources", {})
        remaining_fire = idle.get("fire_units", 0)
        remaining_amb = idle.get("ambulances", 0)
        remaining_pol = idle.get("police", 0)

        allocations = {}
        zone_ids = list(zones.keys())
        # Shuffle to avoid positional bias
        self._rng.shuffle(zone_ids)

        for zone_id in zone_ids:
            z = zones[zone_id]
            has_hazard = (
                z.get("fire", "none") != "none"
                or z.get("patient", "none") not in ("none", "fatal")
                or z.get("traffic", "low") != "low"
            )

            if has_hazard and remaining_fire > 0:
                send_fire = self._rng.randint(0, min(2, remaining_fire))
            else:
                send_fire = 0

            if has_hazard and remaining_amb > 0:
                send_amb = self._rng.randint(0, min(2, remaining_amb))
            else:
                send_amb = 0

            do_traffic = (
                has_hazard
                and remaining_pol > 0
                and self._rng.random() < 0.3
            )

            remaining_fire -= send_fire
            remaining_amb -= send_amb
            if do_traffic:
                remaining_pol -= 1

            allocations[zone_id] = {
                "dispatch_fire": send_fire,
                "dispatch_ambulance": send_amb,
                "control_traffic": do_traffic,
            }

        return {"allocations": allocations}


# ---------------------------------------------------------------------------
# Agent 2: Heuristic Agent
# ---------------------------------------------------------------------------

class HeuristicAgent(BaseAgent):
    """Priority-based dispatch using the exact reward-function mathematics.
    
    Algorithm:
      1. Compute a severity score for each zone:
            S(z) = FIRE_RANK[z.fire] × 10 + PATIENT_RANK[z.patient] × 8 + TRAFFIC_RANK[z.traffic] × 3
      2. Sort zones by S(z) in descending order (most critical first).
      3. For each zone, compute the minimum required fire units and ambulances
         using the exact formulas from env/reward.py:
            R_fire(z) = _get_required_fire(z.fire, weather)
            R_amb(z)  = _get_required_ambulance(z.patient)
      4. Dispatch min(R_fire, idle_fire) fire units and min(R_amb, idle_amb) ambulances.
      5. Deploy police for traffic control if the zone has HEAVY/GRIDLOCK traffic.
      6. Emit a broadcast message when any zone has HIGH+ fire or CRITICAL patients.
    
    This agent demonstrates competent play WITHOUT using an LLM. It should
    score in the range [0.40, 0.75] across all tasks, proving that the
    environment rewards domain-correct resource allocation.
    """
    name = "Heuristic"

    TRAFFIC_RANK = {"low": 0, "heavy": 1, "gridlock": 2}

    def get_action(self, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
        zones = obs.get("zones", {})
        weather = obs.get("weather", "clear")
        idle = obs.get("idle_resources", {})
        remaining_fire = idle.get("fire_units", 0)
        remaining_amb = idle.get("ambulances", 0)
        remaining_pol = idle.get("police", 0)

        # Step 1: Compute per-zone severity score
        zone_priorities: List[Tuple[float, str]] = []
        for zone_id, z in zones.items():
            fire_rank = FIRE_RANK.get(z.get("fire", "none"), 0)
            pat_rank = PATIENT_RANK.get(z.get("patient", "none"), 0)
            traf_rank = self.TRAFFIC_RANK.get(z.get("traffic", "low"), 0)
            score = fire_rank * 10 + pat_rank * 8 + traf_rank * 3
            zone_priorities.append((score, zone_id))

        # Step 2: Sort by severity (descending) — most critical zones first
        zone_priorities.sort(reverse=True)

        allocations = {}
        broadcast_needed = False
        critical_zone = None

        for _, zone_id in zone_priorities:
            z = zones[zone_id]
            fire_level = z.get("fire", "none")
            patient_level = z.get("patient", "none")
            traffic_level = z.get("traffic", "low")

            # Step 3: Compute exact minimum requirements
            req_fire = get_required_fire(fire_level, weather)
            req_amb = get_required_ambulance(patient_level)

            # Step 4: Dispatch the minimum required, capped at available
            send_fire = min(req_fire, remaining_fire)
            send_amb = min(req_amb, remaining_amb)

            # Step 5: Traffic control for congested zones
            do_traffic = (
                traffic_level in ("heavy", "gridlock")
                and remaining_pol > 0
            )

            remaining_fire -= send_fire
            remaining_amb -= send_amb
            if do_traffic:
                remaining_pol -= 1

            allocations[zone_id] = {
                "dispatch_fire": send_fire,
                "dispatch_ambulance": send_amb,
                "control_traffic": do_traffic,
            }

            # Step 6: Check if broadcast is needed
            if fire_level in ("high", "catastrophic") or patient_level == "critical":
                broadcast_needed = True
                critical_zone = zone_id

        action: Dict[str, Any] = {"allocations": allocations}

        # Generate broadcast message when required
        if broadcast_needed and critical_zone:
            z = zones[critical_zone]
            hazard_type = "fire" if FIRE_RANK.get(z.get("fire", "none"), 0) >= 3 else "medical emergency"
            action["public_broadcast_message"] = (
                f"WARNING: {critical_zone} has a severe {hazard_type}. "
                f"All residents must evacuate the area immediately."
            )

        return action


# ---------------------------------------------------------------------------
# Episode runner (agent-agnostic)
# ---------------------------------------------------------------------------

def run_episode(agent: BaseAgent, task_id: int, seed: int) -> EpisodeResult:
    """Run a single episode using the HTTP API and return the result.
    
    Communicates with the server via POST /reset and POST /step.
    """
    agent.reset()

    # Reset environment
    reset_resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=10,
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    rewards: List[float] = []
    step_count = 0
    done = False
    final_score = 0.0
    final_efficiency = 0.0

    while not done:
        step_count += 1
        action = agent.get_action(obs, step_count)

        step_resp = requests.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=15,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()

        obs = step_data["observation"]
        reward = float(step_data["reward"])
        done = step_data.get("done", False)
        info = step_data.get("info", {})
        rewards.append(reward)

        if done:
            final_score = float(info.get("score", 0.0))
            final_efficiency = float(info.get("efficiency", 0.0))

    return EpisodeResult(
        task_id=task_id,
        seed=seed,
        agent_name=agent.name,
        score=final_score,
        efficiency=final_efficiency,
        total_reward=sum(rewards),
        steps=step_count,
    )


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def run_benchmark(n_seeds: int = N_SEEDS, include_llm: bool = False) -> None:
    """Execute the full benchmark suite and print results."""

    # Verify server is running
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=5)
        health.raise_for_status()
        print(f"✅ Server is healthy at {ENV_URL}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Server not reachable at {ENV_URL}: {e}", file=sys.stderr)
        print("   Start the server first:  uvicorn server.app:app --port 7860", file=sys.stderr)
        sys.exit(1)

    agents: List[BaseAgent] = [RandomAgent(), HeuristicAgent()]

    if include_llm:
        try:
            # Import the actual LLM agent from inference.py
            # This requires API credentials to be configured
            from inference import LLMAgent as InferenceLLMAgent
            agents.append(_LLMAgentWrapper())
            print("✅ LLM agent loaded (requires API credentials)", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  LLM agent not available: {e}", file=sys.stderr)

    results: List[EpisodeResult] = []
    total_runs = len(agents) * len(TASKS) * n_seeds

    print(f"\n{'='*80}", file=sys.stderr)
    print(f" BENCHMARK: {len(agents)} agents × {len(TASKS)} tasks × {n_seeds} seeds = {total_runs} episodes", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)

    run_idx = 0
    for agent in agents:
        for task_id in TASKS:
            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx  # Deterministic seed sequence
                run_idx += 1

                try:
                    result = run_episode(agent, task_id, seed)
                    results.append(result)
                    print(
                        f"  [{run_idx:3d}/{total_runs}] {agent.name:12s} | Task {task_id} | "
                        f"seed={seed:5d} | score={result.score:.4f} | "
                        f"eff={result.efficiency:.4f} | steps={result.steps:2d}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"  [{run_idx:3d}/{total_runs}] {agent.name:12s} | Task {task_id} | "
                        f"seed={seed:5d} | ERROR: {e}",
                        file=sys.stderr,
                    )

    # ---------------------------------------------------------------------------
    # Statistical summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*80}", file=sys.stderr)
    print(f" RESULTS SUMMARY", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)

    # Compute per-(agent, task) statistics
    import statistics

    print(f"{'Agent':>12s} | {'Task':>35s} | {'Score (μ±σ)':>15s} | {'Efficiency (μ±σ)':>18s} | {'N':>3s}")
    print(f"{'-'*12}-+-{'-'*35}-+-{'-'*15}-+-{'-'*18}-+-{'-'*3}")

    for agent in agents:
        for task_id in TASKS:
            task_results = [
                r for r in results
                if r.agent_name == agent.name and r.task_id == task_id
            ]
            if not task_results:
                continue

            scores = [r.score for r in task_results]
            efficiencies = [r.efficiency for r in task_results]

            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
            mean_eff = statistics.mean(efficiencies)
            std_eff = statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0.0

            print(
                f"{agent.name:>12s} | {TASK_NAMES[task_id]:>35s} | "
                f"{mean_score:.3f} ± {std_score:.3f} | "
                f"{mean_eff:.3f} ± {std_eff:.3f}    | "
                f"{len(task_results):3d}"
            )

    # Print the results in a format suitable for README
    print(f"\n{'='*80}")
    print(f" README BASELINE TABLE (copy-paste)")
    print(f"{'='*80}\n")
    print(f"| Task | Evaluation Tier | Agent / Policy | Grader Score | Efficiency Score |")
    print(f"| :--- | :--- | :--- | :---: | :---: |")

    for task_id in TASKS:
        for agent in agents:
            task_results = [
                r for r in results
                if r.agent_name == agent.name and r.task_id == task_id
            ]
            if not task_results:
                continue

            scores = [r.score for r in task_results]
            efficiencies = [r.efficiency for r in task_results]
            mean_score = statistics.mean(scores)
            mean_eff = statistics.mean(efficiencies)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

            tier = "Random Baseline" if agent.name == "Random" else "Heuristic Baseline" if agent.name == "Heuristic" else "Reference LLM"
            task_label = {1: "Task 1 (Easy)", 2: "Task 2 (Med)", 3: "Task 3 (Hard)"}[task_id]

            print(
                f"| **{task_label}** | {tier} | {agent.name} Agent | "
                f"{mean_score:.3f} ± {std_score:.3f} | {mean_eff:.3f} |"
            )


# ---------------------------------------------------------------------------
# LLM Agent Wrapper (optional — uses inference.py's LLMAgent via HTTP)
# ---------------------------------------------------------------------------

class _LLMAgentWrapper(BaseAgent):
    """Wraps the inference.py LLM agent for benchmark comparison.
    
    Requires HF_TOKEN or API_KEY to be set in environment variables.
    Uses the same HTTP API as the other benchmark agents.
    """
    name = "LLM"

    def __init__(self):
        from inference import LLMAgent
        self._agent = LLMAgent()

    def reset(self) -> None:
        self._agent.reset_history()

    def get_action(self, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
        from env.models import Observation
        obs_obj = Observation(**obs)
        action, _ = self._agent.get_action(obs_obj, step)
        return action.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple agent types against the Crisis Management Environment."
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of seeds per (agent, task) pair (default: {N_SEEDS}).",
    )
    parser.add_argument(
        "--include-llm", action="store_true",
        help="Include the LLM agent (requires HF_TOKEN or API_KEY).",
    )
    parser.add_argument(
        "--url", type=str, default=ENV_URL,
        help=f"Environment server URL (default: {ENV_URL}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ENV_URL = args.url
    run_benchmark(n_seeds=args.seeds, include_llm=args.include_llm)
