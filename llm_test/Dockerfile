FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed for scientific Python wheels on Debian slim.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY llm_test/requirements.txt /app/requirements.txt

# Install Python dependencies (CPU builds of torch by default).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY llm_test/ /app/llm_test
WORKDIR /app/llm_test

# Default command drops you into the project folder.
CMD ["bash"]
