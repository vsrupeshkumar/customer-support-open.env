"""
inference.py - Baseline agent runner for the Customer Support AI Training Environment.

Implements a simple rule-based policy that deterministically selects actions
based on the current observation. The script:
  1. Initialises the environment.
  2. Runs through all three defined tasks.
  3. Grades each episode using the grader module.
  4. Prints individual task scores and the final weighted average score.

Usage
-----
    python inference.py

The script is fully deterministic and reproducible (fixed seed + fixed policy).
"""

import random
from typing import Dict, List, Optional

from environment import CustomerSupportEnv
from grader import compute_final_score, grade
from tasks import TASKS, Task


# ---------------------------------------------------------------------------
# Rule-based policy
# ---------------------------------------------------------------------------


class RuleBasedAgent:
    """
    A simple deterministic rule-based agent for the customer support environment.

    Policy logic
    ------------
    1. If the issue hasn't been classified yet → classify_issue.
    2. If more info hasn't been requested yet and the task is hard → ask_for_more_info.
    3. If escalation is needed (inferred from observation) → escalate_to_human.
    4. If not resolved and issue classified → respond_with_solution.
    5. If resolved or escalated → mark_resolved.
    """

    def __init__(self, task: Task) -> None:
        self._task = task
        self._asked_for_info = False

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self._asked_for_info = False

    def act(self, observation: Dict) -> str:
        """
        Choose an action based on the current observation.

        Parameters
        ----------
        observation : dict
            The observation dict returned by the environment.

        Returns
        -------
        str
            One of the valid ACTIONS strings.
        """
        classified: bool = observation.get("issue_classified", False)
        resolved: bool = observation.get("resolved", False)
        escalated: bool = observation.get("escalated", False)
        sentiment: str = observation.get("sentiment", "neutral")
        customer_type: str = observation.get("customer_type", "regular")
        history: List = observation.get("conversation_history", [])

        # Step 1 – Always classify first
        if not classified:
            return "classify_issue"

        # Step 2 – Ask for more info on hard task (once)
        if (
            self._task.difficulty == "hard"
            and not self._asked_for_info
            and not resolved
            and not escalated
        ):
            self._asked_for_info = True
            return "ask_for_more_info"

        # Step 3 – If already resolved or escalated, mark resolved
        if resolved or escalated:
            return "mark_resolved"

        # Step 4 – Decide escalation vs direct response
        # Escalate for angry premium customers or if task metadata says so
        needs_escalation = self._task.metadata.get("requires_escalation", False)
        if needs_escalation or (sentiment == "angry" and customer_type == "premium"):
            return "escalate_to_human"

        # Step 5 – Default: provide solution
        return "respond_with_solution"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_task(env: CustomerSupportEnv, task: Task) -> float:
    """
    Run a single task episode with the rule-based agent and return the score.

    Parameters
    ----------
    env  : CustomerSupportEnv
    task : Task

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    agent = RuleBasedAgent(task)
    agent.reset()

    observation = env.reset(scenario_index=task.scenario_index)

    print(f"\n{'=' * 60}")
    print(f"Task: {task.name}  [{task.difficulty.upper()}]")
    print(f"Scenario: {observation['customer_message']}")
    print(f"Sentiment: {observation['sentiment']}  |  Customer: {observation['customer_type']}")
    print(f"{'=' * 60}")

    done = False
    step = 0

    while not done and step < task.max_steps:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        step += 1

        print(
            f"  Step {step:>2} | Action: {action:<25} | "
            f"Reward: {reward:+.2f} | {info.get('feedback', '')}"
        )

    env_state = env.state()
    score = grade(env_state, task)

    print(f"\n  Final State: classified={env_state['issue_classified']} | "
          f"resolved={env_state['resolved']} | escalated={env_state['escalated']}")
    print(f"  Task Score: {score:.4f}  (pass threshold: {task.pass_threshold})")
    passed = score >= task.pass_threshold
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return score


def main() -> None:
    """Entry point – run all tasks and print the final score summary."""
    print("\nCustomer Support AI Training Environment – Baseline Inference")
    print("=" * 60)

    # Fixed seed for full reproducibility
    env = CustomerSupportEnv(seed=42)

    task_scores: Dict[str, float] = {}

    for task in TASKS:
        score = run_task(env, task)
        task_scores[task.task_id] = score

    # Print summary
    final = compute_final_score(task_scores)

    print("\n" + "=" * 60)
    print("SCORE SUMMARY")
    print("=" * 60)
    for task in TASKS:
        tid = task.task_id
        s = task_scores[tid]
        bar = "#" * int(s * 20)
        print(f"  {task.name:<30} [{task.difficulty:<6}]  {s:.4f}  |{bar:<20}|")
    print("-" * 60)
    print(f"  Final Weighted Average Score:           {final:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
