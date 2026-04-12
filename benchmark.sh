#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — Phase 3 Baseline Variance Proof
# =============================================================================
#
# Launches the environment server, waits for readiness, runs the benchmark
# suite (Random + Heuristic agents across all tasks × 10 seeds), then
# reports comparative scores to stdout.
#
# Usage:
#   chmod +x benchmark.sh
#   ./benchmark.sh              # Random + Heuristic (no API key needed)
#   ./benchmark.sh --include-llm  # Also run the LLM agent (needs HF_TOKEN)
#   ./benchmark.sh --seeds 20     # Custom number of seeds
#
# Requirements:
#   - Python 3.10+ with dependencies installed (pip install -r requirements.txt)
#   - Port 7860 available (or set ENV_URL)
# =============================================================================

set -euo pipefail

# Configuration
PORT="${PORT:-7860}"
ENV_URL="${ENV_URL:-http://localhost:${PORT}}"
MAX_WAIT=30  # seconds to wait for server readiness

# Resolve script directory for relative imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect Python: prefer project venv, fall back to system python3
if [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PYTHON="${SCRIPT_DIR}/venv/bin/python3"
elif [ -x "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
else
    PYTHON="$(command -v python3 || command -v python)"
fi
echo "Using Python: ${PYTHON}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Adaptive Crisis Management Environment — Benchmark Suite"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Step 1: Check if server is already running
# ─────────────────────────────────────────────────────────────────────
SERVER_PID=""
SERVER_STARTED=false

if curl -sf "${ENV_URL}/health" > /dev/null 2>&1; then
    echo "✅ Server already running at ${ENV_URL}"
else
    echo "⏳ Starting server on port ${PORT}..."
    "${PYTHON}" -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT}" \
        --log-level warning &
    SERVER_PID=$!
    SERVER_STARTED=true

    # Wait for server readiness with exponential backoff
    echo -n "   Waiting for readiness"
    elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        if curl -sf "${ENV_URL}/health" > /dev/null 2>&1; then
            echo " ✅ (${elapsed}s)"
            break
        fi
        echo -n "."
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ $elapsed -ge $MAX_WAIT ]; then
        echo " ❌ TIMEOUT (${MAX_WAIT}s)"
        echo "   Server failed to start. Check logs above."
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step 2: Run the benchmark
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Running benchmark agents..."
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Pass through any CLI arguments (--include-llm, --seeds N, etc.)
"${PYTHON}" benchmark.py --url "${ENV_URL}" "$@"
BENCHMARK_EXIT=$?

# ─────────────────────────────────────────────────────────────────────
# Step 3: Cleanup
# ─────────────────────────────────────────────────────────────────────
if [ "$SERVER_STARTED" = true ] && [ -n "$SERVER_PID" ]; then
    echo ""
    echo "🛑 Stopping server (PID: ${SERVER_PID})..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
fi

echo ""
if [ $BENCHMARK_EXIT -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  ✅ Benchmark complete. Copy the README table above into README.md"
    echo "═══════════════════════════════════════════════════════════════════"
else
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  ❌ Benchmark failed with exit code ${BENCHMARK_EXIT}"
    echo "═══════════════════════════════════════════════════════════════════"
fi

exit $BENCHMARK_EXIT
