#!/bin/bash

# Run Docker container with Orchestra service
# This exposes port 7000 for the orchestra service in addition to individual services
# It will run container processes as the calling user (UID:GID) so files created inside
# the mounted host folder have the same ownership on the host. This avoids permission
# mismatches between host and container.

# If you prefer to run docker without sudo, add your user to the docker group:
#   sudo groupadd docker || true
#   sudo usermod -aG docker "$USER"
# Then log out and back in (or run `newgrp docker`). Use the script without sudo afterwards.

# Determine user mapping; evaluated on the host when the script runs
# HOST_UID=$(id -u)
# HOST_GID=$(id -g)

sudo docker run -it --rm --gpus all \
  -p 7000:7000 \
  -p 7001:7001 \
  -p 7002:7002 \
  -p 7003:7003 \
  -p 11434:11434 \
  -v "$(pwd)/server:/workspace" \
  -u "${HOST_UID}:${HOST_GID}" \
  -e HOST_UID="${HOST_UID}" -e HOST_GID="${HOST_GID}" \
  extselfvr:latest

# Port mapping:
# 7000 - Orchestra service (main entry point for Unity)
# 7001 - STT service
# 7002 - LLM service
# 7003 - TTS service
# 11434 - Ollama service

# Notes:
# - Running as -u "${HOST_UID}:${HOST_GID}" ensures files created under /workspace
#   (the mounted host directory) are owned by your host user.
# - If your image expects to run as a named user (not root), ensure the image can
#   run processes as an arbitrary numeric UID. Most Python apps will run fine.