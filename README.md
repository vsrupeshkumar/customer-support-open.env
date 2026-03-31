# Customer Support AI Training Environment

An **OpenEnv-compatible** reinforcement learning environment that simulates a real-world customer support system. An AI agent interacts with incoming customer queries and learns to respond appropriately through step-by-step actions.

---

## 📦 Project Structure

```
customer-support-open.env/
├── environment.py   # Core environment logic (reset / step / state)
├── tasks.py         # Task definitions (easy / medium / hard)
├── grader.py        # Deterministic grading and scoring logic
├── inference.py     # Baseline rule-based agent runner
├── openenv.yaml     # Metadata and configuration
├── requirements.txt # Python dependencies
├── Dockerfile       # Containerisation
└── README.md        # This file
```

---

## 🚀 Quick Start

### Run locally (Python 3.10+)

```bash
# Install dependencies (all optional; core uses stdlib only)
pip install -r requirements.txt

# Run the baseline inference script
python inference.py
```

### Run with Docker

```bash
docker build -t customer-support-env .
docker run --rm customer-support-env
```

---

## 🌐 OpenEnv Interface

The environment follows the standard OpenEnv interface:

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `reset(scenario_index=None) -> Dict` | Start a new episode |
| `step` | `step(action: str) -> (obs, reward, done, info)` | Execute one action |
| `state` | `state() -> Dict` | Inspect full internal state |

### Quick example

```python
from environment import CustomerSupportEnv

env = CustomerSupportEnv(seed=42)
obs = env.reset(scenario_index=0)

print(obs["customer_message"])    # customer query
print(obs["sentiment"])           # angry / neutral / happy
print(obs["customer_type"])       # premium / regular

obs, reward, done, info = env.step("classify_issue")
print(reward, info["feedback"])
```

---

## 🎮 Action Space

| Action | Description |
|--------|-------------|
| `classify_issue` | Identify the type of customer issue |
| `respond_with_solution` | Provide a direct resolution |
| `ask_for_more_info` | Request additional details from the customer |
| `escalate_to_human` | Hand off to a human support agent |
| `mark_resolved` | Close the conversation |

---

## 📊 Observation Space

Each observation is a Python `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `customer_message` | `str` | The customer's query text |
| `sentiment` | `str` | `angry`, `neutral`, or `happy` |
| `customer_type` | `str` | `premium` or `regular` |
| `issue_type` | `str` | Ground-truth issue category |
| `conversation_history` | `list` | Log of all agent actions this episode |
| `step` | `int` | Current step number |
| `issue_classified` | `bool` | Whether the agent has classified the issue |
| `resolved` | `bool` | Whether the issue is resolved |
| `escalated` | `bool` | Whether the issue has been escalated |
| `available_actions` | `list` | Valid action strings |

---

## 🏆 Tasks

### Task 1 – Issue Classification (Easy)
- **Goal:** Call `classify_issue` as the first action.
- **Max steps:** 3
- **Pass threshold:** 0.5

### Task 2 – Action Selection (Medium)
- **Goal:** Classify the issue, then choose the correct resolution path (escalate vs. respond directly).
- **Max steps:** 5
- **Pass threshold:** 0.6

### Task 3 – End-to-End Conversation (Hard)
- **Goal:** Handle a complex issue end-to-end: classify → ask for info → escalate → mark resolved.
- **Max steps:** 7
- **Pass threshold:** 0.7

---

## 💰 Reward Function

| Action | Condition | Reward |
|--------|-----------|--------|
| `classify_issue` | First classification | +0.3 |
| `classify_issue` | Repeated | −0.1 |
| `respond_with_solution` | After classifying, no escalation needed | +0.5 |
| `respond_with_solution` | Before classifying | −0.2 |
| `respond_with_solution` | Escalation needed but responded anyway | +0.1 |
| `ask_for_more_info` | Any time | +0.1 |
| `escalate_to_human` | Required by scenario | +0.5 (+0.1 premium bonus) |
| `escalate_to_human` | Unnecessary | −0.3 |
| `mark_resolved` | After proper handling | +0.3 |
| `mark_resolved` | Premature | −0.1 to −0.2 |

---

## 📈 Scoring

Each task grader returns a score in `[0.0, 1.0]`. The final score is a **weighted average**:

| Task | Weight |
|------|--------|
| Easy | 0.2 |
| Medium | 0.3 |
| Hard | 0.5 |

---

## 🔧 Resource Requirements

- **CPU:** 2 vCPU
- **RAM:** 8 GB
- **Max execution time:** < 20 minutes
- **Dependencies:** Python standard library only (no GPU required)

---

## 🤗 Deployment to Hugging Face Spaces

This project is backend-only with no frontend code. To deploy:

1. Push this repository to Hugging Face Spaces as a **Docker** space.
2. The `Dockerfile` handles the full setup.
3. The `CMD` runs `inference.py` on startup.

---

## 📝 License

MIT