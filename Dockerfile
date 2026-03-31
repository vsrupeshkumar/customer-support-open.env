# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY environment.py .
COPY tasks.py .
COPY grader.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Default command: run the baseline inference script
CMD ["python", "inference.py"]
