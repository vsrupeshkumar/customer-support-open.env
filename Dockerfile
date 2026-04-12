# ============================================================
# Adaptive Crisis Management Environment — Production Dockerfile
# Hugging Face Spaces compatible (port 7860, non-root user)
# ============================================================

FROM python:3.10-slim

# ---- System-level hardening & metadata -------------------------
LABEL maintainer="Meta Capstone Team" \
      version="4.1.0" \
      description="OpenEnv-compliant Crisis Management RL Environment API"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Working directory -----------------------------------------
WORKDIR /app

# ---- Non-root user (Hugging Face Spaces security requirement) ---
RUN useradd -m -u 1000 appuser

# ---- Python dependencies (installed as root for system-wide access) ----
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application source ----------------------------------------
# Set Difference Architecture: Copy everything, filtered strictly by .dockerignore
COPY --chown=appuser:appuser . .

# ---- Switch to non-root user -----------------------------------
USER 1000

# ---- Network ---------------------------------------------------
EXPOSE 7860

# ---- Runtime ---------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Hugging Face Spaces expects the service to bind on 0.0.0.0:7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
