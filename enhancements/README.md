# Parallel Intelligence & Visualization Layer

This is an optional, zero-risk enhancement layer for the **Adaptive Crisis Management Environment (OpenEnv)**. It provides real-time telemetry, a crisis visualization dashboard, non-blocking logging, and an LLM-backed agent assist feature.

**Core Principle**: "Observe, don’t interfere. Enhance, don’t replace."
This layer never touches the core simulation logic, uses read-only state copies, and automatically disables itself if it detects any performance degradation.

## 🚀 Features

1. **Real-Time Telemetry Dashboard**: A lightweight UI to monitor the simulation state, reward trajectories, and crisis maps without affecting the backend.
2. **Decision Intelligence (Agent Assist)**: Provides asynchronous, LLM-generated explanations and better action suggestions for agent evaluations.
3. **Middleware Logging**: Captures request/response metrics cleanly to a local JSON lines file.
4. **Performance Guardrails**: Continuously monitors endpoint latency and automatically disables enhancements if they introduce overhead beyond a set threshold.

## ⚙️ Configuration Flags

Configure the enhancements using environment variables.

| Flag | Default | Description |
| :--- | :---: | :--- |
| `ENABLE_ENHANCEMENTS` | `true` | Master switch. If `false`, the entire layer is ignored. |
| `ENABLE_TELEMETRY` | `true` | Enables the in-memory telemetry store and `/enhanced/metrics`. |
| `ENABLE_VISUALIZATION` | `true` | Mounts the interactive dashboard at `/enhanced/ui`. |
| `ENABLE_AGENT_ASSIST` | `false` | Enables the LLM evaluation overlay at `/enhanced/explain`. |
| `ENABLE_ENHANCED_LOGGING` | `true` | Enables writing async metrics to the configured log file. |
| `ENHANCED_LOG_PATH` | `/tmp/enhanced_logs.json` | Path where non-blocking JSONL metrics are saved. |
| `ENHANCEMENT_MAX_LATENCY_MS` | `1200` | Hard latency cap. Exceeding this disables enhancements. |
| `ENHANCEMENT_LATENCY_RATIO` | `1.05` | Max allowed overhead compared to initial baseline (e.g., 5%). |

*Note: Agent Assist requires API credentials (`HF_TOKEN`, `API_KEY`, or `GROQ_API_KEY`) to be set in the environment, utilizing the same setup as `inference.py`.*

## 🌐 Endpoints

If enabled, the following endpoints are appended to the FastAPI application:

- **`GET /enhanced/ui`**
  Loads the interactive dashboard to view the map, global metrics, and active state payload.

- **`GET /enhanced/metrics?session_id=<uuid>`**
  Returns the complete internal timeline, state, action payload, and metrics snapshot for a specific session (or `"default"`).

- **`POST /enhanced/explain`**
  Submits an action and state for asynchronous LLM evaluation.
  *Returns:* `{"request_id": "<uuid>", "status": "queued"}`

- **`GET /enhanced/explain/{request_id}`**
  Retrieves the result of the asynchronous evaluation.
  *Returns:* `{"status": "ready", "result": {"verdict": "...", "rationale": "...", ...}}`
