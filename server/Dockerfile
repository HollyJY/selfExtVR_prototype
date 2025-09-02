# 1) Base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/hf \
    TRANSFORMERS_CACHE=/workspace/.cache/hf \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/hf
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git curl supervisor && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

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

EXPOSE 7001 7002 7003

CMD ["/usr/bin/supervisord", "-c", "/workspace/supervisor.conf"]
