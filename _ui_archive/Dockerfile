FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY . .
RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="https://openrouter.ai/api/v1"
ENV MODEL_NAME="meta-llama/llama-3-8b-instruct:free"
ENV HF_TOKEN=""

EXPOSE 7860

CMD ["uv", "run", "server"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
