#!/bin/bash
set -e

echo "🚨 [DEVOPS] Initializing Guillotine-Proof Sync Pipeline..."

# ==============================================================================
# 1. THE DUAL-REMOTE CONFIGURATION
# ==============================================================================
echo "[1/3] Configuring explicit remotes to prevent drift..."

# Safely rename 'origin' if it exists to avoid collisions
if git remote | grep -q "^origin$"; then
    git remote rename origin github
fi

# Ensure both remotes exist cleanly
git remote remove github 2>/dev/null || true
git remote add github https://github.com/vsrupeshkumar/customer-support-open.env.git

git remote remove huggingface 2>/dev/null || true
git remote add huggingface "https://Anbu-00001:${HF_TOKEN}@huggingface.co/spaces/Anbu-00001/adaptive-crisis-env"

echo "Remotes configured:"
git remote -v

# ==============================================================================
# 2. THE ATOMIC COMMIT
# ==============================================================================
echo "[2/3] Preparing atomic commit..."

git add openenv.yaml server/app.py README.md

if git diff --cached --quiet; then
    echo "No modifications detected. Skipping atomic commit."
else
    git commit -m "feat: implement resilient POMDP /reset handler (Fix 422)
fix: add OpenEnv spec_version 1.0 to manifest
docs: inject YAML frontmatter for HF Docker SDK compliance"
    echo "Atomic commit executed successfully."
fi

# ==============================================================================
# 3. PRE-VALIDATION GATE
# ==============================================================================
echo "[3/4] Executing 3-Stage Pre-Validation Gate..."

HF_PING_URL=${1:-"https://anbu-00001-adaptive-crisis-env.hf.space"}

# Stage 0: Mandatory execution boundary check
if [[ ! -f "openenv.yaml" ]] || [[ ! -f "Dockerfile" ]]; then
    echo "❌ CRITICAL ERROR: Environment validation failed! Missing openenv.yaml or Dockerfile."
    echo "Aborting deployment to avoid standard OpenEnv Guillotine Failure."
    exit 1
fi

# Stage 1: Ping HF Space via /reset
echo "-> Stage 1/3: Pinging HF Space (${HF_PING_URL}/reset)..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${HF_PING_URL}/reset" -H "Content-Type: application/json" -d '{}' || echo "000")
if [ "$HTTP_STATUS" != "200" ]; then
    echo "❌ CRITICAL ERROR: Stage 1 Failed. /reset endpoint returned HTTP $HTTP_STATUS."
    echo "Deployment aborted."
    exit 1
fi
echo "✅ Stage 1 passed."

# Stage 2: Local Docker Build check
echo "-> Stage 2/3: Validating Local Docker Build..."
if ! docker build -t adaptive-crisis-env:validation . ; then
    echo "❌ CRITICAL ERROR: Stage 2 Failed. Local Docker build encountered errors."
    echo "Deployment aborted."
    exit 1
fi
echo "✅ Stage 2 passed."

# Stage 3: openenv validate check
echo "-> Stage 3/3: Executing openenv validity check..."
# source virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if ! openenv validate . ; then
    echo "❌ CRITICAL ERROR: Stage 3 Failed. 'openenv validate .' encountered errors."
    echo "Deployment aborted."
    exit 1
fi
echo "✅ Stage 3 passed."

echo "🚀 VALIDATION PASSED: Initiating Deployment..."

# ==============================================================================
# 4. THE DEPLOYMENT PIPELINE
# ==============================================================================
echo "[4/4] Executing push to remote tiers..."

# Push parallel state to both instances (falling back to master if main doesn't exist)
echo "Deploying to GitHub [Open-Source Tier]..."
git push github main || git push github master

echo "Deploying to Hugging Face [Evaluation Execution Tier]..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN is not set in the environment. Attempting unauthenticated / cached push. If this fails, export HF_TOKEN and rerun."
fi
git push huggingface main || git push huggingface master

echo "✅ [DEVOPS] Master Sync Completed. The Adaptive Crisis Environment is deployed."
