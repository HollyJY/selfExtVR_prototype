# 1) Base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/hf \
    TRANSFORMERS_CACHE=/workspace/.cache/hf \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/hf \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/workspace/models/llama \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git curl supervisor ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

## 2) Install Ollama (for local LLM serving)
# Note: Requires network at build time. Installs /usr/local/bin/ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Ensure model cache directory exists (and is mountable as a volume if desired)
RUN mkdir -p ${OLLAMA_MODELS}

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
    && pip3 install --ignore-installed -r requirements.txt

## IndexTTS (local) installation
# Copy the IndexTTS project into the image and install it so the TTS service can import it.
COPY models/indexTTS/ ./models/indexTTS/
RUN pip3 install --no-cache-dir -e ./models/indexTTS

COPY services/ ./services/
COPY config/ ./config/
RUN mkdir -p data/sessions data/exports logs

COPY supervisor.conf ./
COPY docker-entrypoint.sh ./
RUN chmod +x /workspace/docker-entrypoint.sh

EXPOSE 7001 7002 7003 11434

ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/workspace/supervisor.conf"]
