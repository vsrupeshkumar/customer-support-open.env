# ============================================================
# Adaptive Crisis Management Environment — Production Dockerfile
# Hugging Face Spaces compatible (port 7860, non-root user)
# ============================================================

FROM python:3.10-slim

# ---- System-level hardening & metadata -------------------------
LABEL maintainer="Meta Capstone Team" \
      version="4.0.0" \
      description="OpenEnv-compliant Crisis Management RL Environment API"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Working directory -----------------------------------------
WORKDIR /app

# ---- Non-root user (Hugging Face Spaces security requirement) ---
# Create a system group + user named "appuser" with no login shell
RUN groupadd --system appuser && \
    useradd --system --gid appuser --no-create-home appuser && \
    chown -R appuser:appuser /app

# ---- Python dependencies (installed as root for system-wide access) ----
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application source ----------------------------------------
# Copy only what's needed at runtime — no venv, no dev tooling
COPY env/           ./env/
COPY inference.py   ./inference.py
COPY server.py      ./server.py
COPY openenv.yaml   ./openenv.yaml
COPY metrics_tracker.py ./metrics_tracker.py

# ---- Transfer ownership to non-root user -----------------------
RUN chown -R appuser:appuser /app

# ---- Switch to non-root user -----------------------------------
USER appuser

# ---- Network ---------------------------------------------------
EXPOSE 7860

# ---- Runtime ---------------------------------------------------
# Hugging Face Spaces expects the service to bind on 0.0.0.0:7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
