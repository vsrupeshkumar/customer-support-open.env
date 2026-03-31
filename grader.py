"""
grader.py - Grading and scoring logic for the Customer Support AI Training Environment.

Each task has a dedicated deterministic grader that returns a score in [0.0, 1.0].
The final score is a weighted average across all tasks.

Grading philosophy
------------------
- Scores are always in the range [0.0, 1.0].
- Partial credit is given for partially correct behaviour.
- Penalties reduce the score but are clamped at 0.0 (no negative totals).
- The grader inspects the environment's state() snapshot after an episode ends.
"""

from typing import Any, Dict, List

from tasks import Task


# ---------------------------------------------------------------------------
# Individual task graders
# ---------------------------------------------------------------------------


def grade_task_easy(env_state: Dict[str, Any], task: Task) -> float:
    """
    Task 1 (Easy) Grader – Issue Classification.

    Scoring breakdown
    -----------------
    +0.5  : issue was classified at all
    +0.3  : classification happened within the first 2 steps
    +0.2  : no incorrect actions taken (clean run)
    –0.1  : deducted per repeated/incorrect action (capped at –0.3)
    """
    score = 0.0
    history: List[Dict[str, Any]] = env_state.get("history", [])
    incorrect = env_state.get("incorrect_actions", 0)

    # Was the issue ever classified?
    if env_state.get("issue_classified"):
        score += 0.5

        # Was it the first or second action?
        classified_step = next(
            (i for i, h in enumerate(history) if h["action"] == "classify_issue"),
            None,
        )
        if classified_step is not None and classified_step < 2:
            score += 0.3

    # Clean run bonus
    if incorrect == 0:
        score += 0.2

    # Penalty for each incorrect action
    penalty = min(incorrect * 0.1, 0.3)
    score = max(0.0, score - penalty)

    return round(min(score, 1.0), 4)


def grade_task_medium(env_state: Dict[str, Any], task: Task) -> float:
    """
    Task 2 (Medium) Grader – Action Selection.

    Scoring breakdown
    -----------------
    +0.3  : issue classified
    +0.4  : correct escalation/response decision taken
    +0.2  : issue marked resolved at the end
    –0.15 : per unnecessary or incorrect action (capped at –0.4)
    """
    score = 0.0
    history: List[Dict[str, Any]] = env_state.get("history", [])
    actions_taken = [h["action"] for h in history]
    incorrect = env_state.get("incorrect_actions", 0)

    requires_escalation: bool = task.metadata.get("requires_escalation", False)

    if env_state.get("issue_classified"):
        score += 0.3

    # Did the agent make the right resolution decision?
    if requires_escalation:
        if "escalate_to_human" in actions_taken:
            score += 0.4
        elif "respond_with_solution" in actions_taken:
            # Wrong path – gave direct response when escalation was needed
            score += 0.1
    else:
        if "respond_with_solution" in actions_taken:
            score += 0.4
        elif "escalate_to_human" in actions_taken:
            score += 0.1

    # Resolution bonus
    if env_state.get("resolved") or env_state.get("escalated"):
        score += 0.2

    # Penalty
    penalty = min(incorrect * 0.15, 0.4)
    score = max(0.0, score - penalty)

    return round(min(score, 1.0), 4)


def grade_task_hard(env_state: Dict[str, Any], task: Task) -> float:
    """
    Task 3 (Hard) Grader – End-to-End Conversation.

    Scoring breakdown
    -----------------
    +0.2  : issue classified
    +0.1  : at least one 'ask_for_more_info' used (shows thoroughness)
    +0.3  : correct escalation decision (escalate_to_human for this scenario)
    +0.2  : mark_resolved called after proper handling
    +0.1  : efficient resolution (total steps ≤ 5)
    –0.1  : per incorrect action (capped at –0.4)
    –0.05 : per extra step beyond 5 (capped at –0.2)

    Graders inspect the expected_actions list from the Task descriptor to
    determine whether the agent followed the recommended sequence.
    """
    score = 0.0
    history: List[Dict[str, Any]] = env_state.get("history", [])
    actions_taken = [h["action"] for h in history]
    step_count: int = env_state.get("step_count", 0)
    incorrect: int = env_state.get("incorrect_actions", 0)

    # Classification
    if env_state.get("issue_classified"):
        score += 0.2

    # Thoroughness – asked for more info at some point
    if "ask_for_more_info" in actions_taken:
        score += 0.1

    # Correct escalation
    if env_state.get("escalated"):
        score += 0.3

    # Final resolution
    if "mark_resolved" in actions_taken and (
        env_state.get("resolved") or env_state.get("escalated")
    ):
        score += 0.2

    # Efficiency bonus
    if step_count <= 5:
        score += 0.1

    # Penalties
    penalty = min(incorrect * 0.1, 0.4)
    extra_steps = max(0, step_count - 5)
    step_penalty = min(extra_steps * 0.05, 0.2)
    score = max(0.0, score - penalty - step_penalty)

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_GRADERS = {
    "task_1_easy": grade_task_easy,
    "task_2_medium": grade_task_medium,
    "task_3_hard": grade_task_hard,
}


def grade(env_state: Dict[str, Any], task: Task) -> float:
    """
    Grade a completed episode for the given task.

    Parameters
    ----------
    env_state : dict
        Snapshot from ``CustomerSupportEnv.state()`` after the episode ends.
    task : Task
        The task descriptor being evaluated.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    grader_fn = _GRADERS.get(task.task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id='{task.task_id}'")
    return grader_fn(env_state, task)


def compute_final_score(task_scores: Dict[str, float]) -> float:
    """
    Compute the weighted average score across all tasks.

    Weights: easy=0.2, medium=0.3, hard=0.5 (harder tasks matter more).

    Parameters
    ----------
    task_scores : dict
        Mapping of task_id -> score.

    Returns
    -------
    float
        Weighted average score in [0.0, 1.0].
    """
    weights = {
        "task_1_easy": 0.2,
        "task_2_medium": 0.3,
        "task_3_hard": 0.5,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for task_id, score in task_scores.items():
        w = weights.get(task_id, 1.0)
        weighted_sum += w * score
        total_weight += w

    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)
