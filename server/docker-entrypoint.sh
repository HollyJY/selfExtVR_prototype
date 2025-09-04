#!/usr/bin/env bash
set -euo pipefail

# Resolve repo workspace
cd /workspace

CFG="config/app.yml"

get_yaml() {
  # Pass arguments to inline Python via stdin filename '-' and "$@"
  python3 - "$@" <<'PY'
import os, sys, yaml
cfg_path = sys.argv[1]
key = sys.argv[2]
def get(d, path):
    cur = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur
try:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
except Exception:
    cfg = {}
val = get(cfg or {}, key)
print(val if val is not None else '')
PY
  }

# Derive models_root (default /workspace/models)
MODELS_ROOT=$(get_yaml "$CFG" paths.models_root)
if [[ -z "${MODELS_ROOT}" ]]; then
  MODELS_ROOT="/workspace/models"
fi

# Derive Ollama models dir: explicit paths.ollama_models_dir or MODELS_ROOT/llama
OLLAMA_MODELS_DIR=$(get_yaml "$CFG" paths.ollama_models_dir)
if [[ -z "${OLLAMA_MODELS_DIR}" ]]; then
  OLLAMA_MODELS_DIR="${MODELS_ROOT}/llama"
fi
mkdir -p "${OLLAMA_MODELS_DIR}"

# Configure HF cache if provided; otherwise keep default
HF_CACHE_DIR=$(get_yaml "$CFG" paths.hf_cache_dir)
if [[ -n "${HF_CACHE_DIR}" ]]; then
  mkdir -p "${HF_CACHE_DIR}"
  export HF_HOME="${HF_CACHE_DIR}"
  export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
  export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}"
fi

# Export Ollama env so supervisord children inherit it
export OLLAMA_MODELS="${OLLAMA_MODELS_DIR}"
export OLLAMA_HOST="0.0.0.0:11434"

echo "[entrypoint] MODELS_ROOT=${MODELS_ROOT}"
echo "[entrypoint] OLLAMA_MODELS=${OLLAMA_MODELS}"
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  echo "[entrypoint] HF_CACHE_DIR=${HF_CACHE_DIR}"
fi

# Ensure Ollama is installed (fallback if image build skipped it)
if ! command -v ollama >/dev/null 2>&1; then
  echo "[entrypoint] Ollama not found; installing..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo "[entrypoint] curl not available; attempting apt-get install curl"
    apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[entrypoint] Failed to install Ollama; continue without it."
  fi
fi

exec "$@"
