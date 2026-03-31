"""
tasks.py - Task definitions for the Customer Support AI Training Environment.

Defines three tasks of increasing difficulty:
  - Task 1 (Easy)   : Classify the issue type correctly.
  - Task 2 (Medium) : Choose the correct action based on the support scenario.
  - Task 3 (Hard)   : Handle an issue end-to-end in a multi-step conversation.

Each task specifies which scenario(s) to use, the maximum allowed steps,
and an expected action sequence for evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """Descriptor for a single evaluation task."""

    task_id: str
    name: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    scenario_index: int  # Index into environment.SCENARIOS
    max_steps: int
    # Ordered list of actions the agent is expected to take (for grading)
    expected_actions: List[str] = field(default_factory=list)
    # Minimum score threshold to consider the task "passed"
    pass_threshold: float = 0.6
    # Additional metadata for the grader
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: List[Task] = [
    # ------------------------------------------------------------------
    # Task 1 – Easy: Identify / classify the incoming issue
    # The agent only needs to call "classify_issue" at the right moment.
    # A single correct classification within 3 steps earns full marks.
    # ------------------------------------------------------------------
    Task(
        task_id="task_1_easy",
        name="Issue Classification",
        difficulty="easy",
        description=(
            "Given a customer message, the agent must correctly identify "
            "and classify the issue type by calling 'classify_issue' as its "
            "first action. Success is measured by whether classification "
            "happens early (step 1 or 2) and by the absence of penalties."
        ),
        scenario_index=0,  # payment_failure scenario
        max_steps=3,
        expected_actions=["classify_issue"],
        pass_threshold=0.5,
        metadata={"target_issue_type": "payment_failure"},
    ),
    # ------------------------------------------------------------------
    # Task 2 – Medium: Choose the correct action sequence
    # The agent must classify and then pick the right resolution path.
    # Scenarios are chosen to test the escalation vs. direct-response
    # decision boundary.
    # ------------------------------------------------------------------
    Task(
        task_id="task_2_medium",
        name="Action Selection",
        difficulty="medium",
        description=(
            "Given a customer scenario that may or may not require escalation, "
            "the agent must: (1) classify the issue, then (2) choose between "
            "'respond_with_solution' or 'escalate_to_human' based on the "
            "scenario context. Choosing the wrong path incurs a penalty."
        ),
        scenario_index=3,  # account_issue requiring escalation
        max_steps=5,
        expected_actions=["classify_issue", "escalate_to_human", "mark_resolved"],
        pass_threshold=0.6,
        metadata={"requires_escalation": True, "customer_type": "premium"},
    ),
    # ------------------------------------------------------------------
    # Task 3 – Hard: Full multi-step conversation management
    # The agent must handle an angry premium customer with a double-charge,
    # navigating through classification → escalation → resolution with
    # appropriate empathy checks at each step.
    # ------------------------------------------------------------------
    Task(
        task_id="task_3_hard",
        name="End-to-End Conversation",
        difficulty="hard",
        description=(
            "Handle a complex, multi-step customer support conversation. "
            "The agent must: (1) classify the issue, (2) optionally ask for "
            "more info, (3) choose the correct resolution or escalation path, "
            "and (4) mark the issue resolved. The scoring penalises incorrect "
            "action order, unnecessary actions, and skipped steps."
        ),
        scenario_index=5,  # double-charge / payment_failure, premium, escalation needed
        max_steps=7,
        expected_actions=[
            "classify_issue",
            "ask_for_more_info",
            "escalate_to_human",
            "mark_resolved",
        ],
        pass_threshold=0.7,
        metadata={
            "requires_escalation": True,
            "customer_type": "premium",
            "sentiment": "angry",
        },
    ),
]


def get_task(task_id: str) -> Optional[Task]:
    """Return a Task by its task_id, or None if not found."""
    return next((t for t in TASKS if t.task_id == task_id), None)
