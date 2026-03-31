"""
environment.py - Core Customer Support AI Training Environment

Implements the OpenEnv-compatible interface with reset(), step(action), and state() methods.
Simulates a realistic customer support system where an AI agent handles customer queries.
"""

import random
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All valid agent actions
ACTIONS: List[str] = [
    "classify_issue",
    "respond_with_solution",
    "ask_for_more_info",
    "escalate_to_human",
    "mark_resolved",
]

# Issue categories the agent must recognize
ISSUE_TYPES: List[str] = [
    "payment_failure",
    "delayed_delivery",
    "refund_request",
    "account_issue",
]

# Sentiment levels derived from message content / random draw
SENTIMENTS: List[str] = ["angry", "neutral", "happy"]

# Customer tiers that affect escalation policy
CUSTOMER_TYPES: List[str] = ["premium", "regular"]

# ---------------------------------------------------------------------------
# Scenario bank - deterministic seed produces reproducible episodes
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict[str, Any]] = [
    {
        "message": (
            "I tried to pay for my order but the payment keeps failing! "
            "This is the third time I'm trying."
        ),
        "issue_type": "payment_failure",
        "sentiment": "angry",
        "customer_type": "regular",
        "requires_escalation": False,
        "solution": (
            "Please clear your browser cache and try again. "
            "If the issue persists, contact your bank."
        ),
    },
    {
        "message": (
            "My package was supposed to arrive 5 days ago but still nothing. "
            "Where is my order?"
        ),
        "issue_type": "delayed_delivery",
        "sentiment": "angry",
        "customer_type": "premium",
        "requires_escalation": False,
        "solution": (
            "We sincerely apologise for the delay. Your order is in transit. "
            "Expected delivery within 2 business days."
        ),
    },
    {
        "message": "I'd like to request a refund for order #12345.",
        "issue_type": "refund_request",
        "sentiment": "neutral",
        "customer_type": "regular",
        "requires_escalation": False,
        "solution": (
            "Your refund has been initiated and will reflect in 5-7 business days."
        ),
    },
    {
        "message": (
            "I cannot log into my account. It says my credentials are invalid "
            "even after resetting my password."
        ),
        "issue_type": "account_issue",
        "sentiment": "neutral",
        "customer_type": "premium",
        "requires_escalation": True,
        "solution": (
            "We are escalating this to our account security team. "
            "You will receive an email within 1 hour."
        ),
    },
    {
        "message": "Hi, just checking - did my refund go through?",
        "issue_type": "refund_request",
        "sentiment": "happy",
        "customer_type": "regular",
        "requires_escalation": False,
        "solution": "Yes, your refund has been successfully processed.",
    },
    {
        "message": (
            "My card was charged twice for the same order! "
            "I need this fixed immediately."
        ),
        "issue_type": "payment_failure",
        "sentiment": "angry",
        "customer_type": "premium",
        "requires_escalation": True,
        "solution": (
            "We sincerely apologise. Your duplicate charge has been identified "
            "and a refund will be processed within 24 hours."
        ),
    },
]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CustomerSupportEnv:
    """
    OpenEnv-compatible Customer Support AI Training Environment.

    The agent interacts with simulated customer queries step-by-step,
    choosing from a fixed action set. The environment tracks conversation
    state and computes shaped rewards at each step.

    Interface
    ---------
    reset(scenario_index) -> observation dict
    step(action)          -> (observation, reward, done, info)
    state()               -> full internal state dict
    """

    # Maximum turns per episode to avoid infinite loops
    MAX_STEPS: int = 10

    def __init__(self, seed: int = 42) -> None:
        """
        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        """
        self._rng = random.Random(seed)
        self._scenario: Dict[str, Any] = {}
        self._history: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._done: bool = True
        self._issue_classified: bool = False
        self._resolved: bool = False
        self._escalated: bool = False
        self._incorrect_actions: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, scenario_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Begin a new episode.

        Parameters
        ----------
        scenario_index : int, optional
            Index into SCENARIOS list. If None, a random scenario is chosen
            using the environment's RNG (for reproducibility).

        Returns
        -------
        observation : dict
            Initial observation for the agent.
        """
        if scenario_index is not None:
            idx = scenario_index % len(SCENARIOS)
        else:
            idx = self._rng.randrange(len(SCENARIOS))

        self._scenario = SCENARIOS[idx].copy()
        self._history = []
        self._step_count = 0
        self._done = False
        self._issue_classified = False
        self._resolved = False
        self._escalated = False
        self._incorrect_actions = 0

        return self._build_observation()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one agent action and advance the environment.

        Parameters
        ----------
        action : str
            One of the valid ACTIONS strings.

        Returns
        -------
        observation : dict
        reward       : float  – shaped reward signal
        done         : bool   – whether the episode has ended
        info         : dict   – auxiliary diagnostic information
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new one.")

        if action not in ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid actions: {ACTIONS}")

        reward, info = self._apply_action(action)
        self._step_count += 1

        # Episode ends when the issue is resolved (via mark_resolved or direct
        # respond_with_solution) or when max steps are reached.
        # Escalation alone does NOT end the episode – the agent must still call
        # mark_resolved to formally close the conversation.
        if self._resolved or self._step_count >= self.MAX_STEPS:
            self._done = True

        observation = self._build_observation()
        return observation, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the full internal state of the environment.

        Useful for debugging and grading.
        """
        return {
            "scenario": self._scenario,
            "history": list(self._history),
            "step_count": self._step_count,
            "done": self._done,
            "issue_classified": self._issue_classified,
            "resolved": self._resolved,
            "escalated": self._escalated,
            "incorrect_actions": self._incorrect_actions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        """Construct the agent's observation from the current state."""
        return {
            "customer_message": self._scenario["message"],
            "sentiment": self._scenario["sentiment"],
            "customer_type": self._scenario["customer_type"],
            "issue_type": self._scenario.get("issue_type", "unknown"),
            "conversation_history": list(self._history),
            "step": self._step_count,
            "issue_classified": self._issue_classified,
            "resolved": self._resolved,
            "escalated": self._escalated,
            "available_actions": ACTIONS,
        }

    def _apply_action(self, action: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process the agent's chosen action and return (reward, info).

        Reward shaping summary
        ----------------------
        - classify_issue      : +0.3 on first correct use, –0.1 if repeated
        - respond_with_solution: +0.4 if issue classified & correct scenario,
                                 –0.2 if used before classifying
        - ask_for_more_info   : +0.1 (small positive; valid but not optimal)
        - escalate_to_human   : +0.5 if scenario requires it, –0.3 if not needed
        - mark_resolved       : +0.3 if properly handled, –0.2 if premature
        Incorrect action repeated: additional –0.1 penalty
        """
        reward = 0.0
        info: Dict[str, Any] = {"action": action}

        if action == "classify_issue":
            reward, info = self._handle_classify(info)

        elif action == "respond_with_solution":
            reward, info = self._handle_respond(info)

        elif action == "ask_for_more_info":
            reward, info = self._handle_ask_more(info)

        elif action == "escalate_to_human":
            reward, info = self._handle_escalate(info)

        elif action == "mark_resolved":
            reward, info = self._handle_mark_resolved(info)

        # Record in conversation history
        self._history.append({"role": "agent", "action": action, "reward": reward})
        info["step"] = self._step_count
        return reward, info

    def _handle_classify(self, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if not self._issue_classified:
            # First classification attempt – always correct (deterministic env)
            self._issue_classified = True
            reward = 0.3
            info["feedback"] = (
                f"Issue correctly classified as '{self._scenario['issue_type']}'."
            )
        else:
            # Repeated classification is a waste of a turn
            reward = -0.1
            self._incorrect_actions += 1
            info["feedback"] = "Issue was already classified. Unnecessary action."
        return reward, info

    def _handle_respond(self, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if not self._issue_classified:
            # Responding before understanding the issue is poor practice
            reward = -0.2
            self._incorrect_actions += 1
            info["feedback"] = (
                "Attempted to respond before classifying the issue. "
                "Partial penalty applied."
            )
        elif self._scenario.get("requires_escalation"):
            # This scenario needs escalation, not a direct response
            reward = 0.1
            info["feedback"] = (
                "Provided a response, but this issue requires escalation. "
                "Consider escalating."
            )
        else:
            # Correct resolution path
            self._resolved = True
            reward = 0.5
            info["feedback"] = "Solution provided and issue resolved."
            info["solution"] = self._scenario["solution"]
        return reward, info

    def _handle_ask_more(self, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        # Asking for more info is always valid but not the most efficient move
        reward = 0.1
        info["feedback"] = (
            "Requested additional information from the customer. "
            "Acceptable but not optimal."
        )
        return reward, info

    def _handle_escalate(self, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if self._scenario.get("requires_escalation"):
            self._escalated = True
            reward = 0.5
            info["feedback"] = (
                "Correctly escalated to human agent. "
                "Bonus for premium/complex issue handling."
            )
            # Bonus for premium customer escalation
            if self._scenario["customer_type"] == "premium":
                reward += 0.1
                info["feedback"] += " Premium customer bonus applied."
        else:
            # Unnecessary escalation wastes resources
            reward = -0.3
            self._incorrect_actions += 1
            info["feedback"] = (
                "Unnecessary escalation. This issue could have been resolved directly."
            )
        return reward, info

    def _handle_mark_resolved(
        self, info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        if self._issue_classified and (self._resolved or self._escalated):
            # Clean resolution after proper handling
            reward = 0.3
            self._resolved = True
            info["feedback"] = "Issue marked as resolved after proper handling. Well done!"
        elif self._step_count == 0:
            # Marking resolved immediately without any work
            reward = -0.2
            self._incorrect_actions += 1
            info["feedback"] = "Premature resolution. No actions were taken first."
        else:
            # Premature resolution without proper handling
            reward = -0.1
            self._incorrect_actions += 1
            self._resolved = True  # episode ends but with penalty
            info["feedback"] = (
                "Issue marked resolved before fully handling it. Partial penalty."
            )
        return reward, info
