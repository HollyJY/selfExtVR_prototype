#!/usr/bin/env bash
# Fire-and-forget runner that survives SSH disconnects when launched with nohup/tmux/screen.
# Environment overrides: ITER_START, ITER_END, INPUT, OUTPUT_DIR, SESSION_ID, TRIAL_ID, SLEEP, LLM_API_URL.
set -euo pipefail

ITER_START=${ITER_START:-1}
ITER_END=${ITER_END:-200}
INPUT=${INPUT:-final_12_scenes.csv}
OUTPUT_DIR=${OUTPUT_DIR:-llm_tables_out}
SESSION_ID=${SESSION_ID:-test_llm}
TRIAL_ID=${TRIAL_ID:-1}
SLEEP=${SLEEP:-0.3}
API_URL=${LLM_API_URL:-http://192.168.37.177:7002/api/v1/llm}
SIMILARITY_OUT=${SIMILARITY_OUT:-llm_tables_similarity_out}

ts() { date +"%Y-%m-%d %H:%M:%S"; }
LOG_FILE=${LOG_FILE:-run_$(date +"%Y%m%d_%H%M%S").log}

echo "[start $(ts)] iter ${ITER_START}-${ITER_END}, log=${LOG_FILE}"

for i in $(seq "$ITER_START" "$ITER_END"); do
  echo "[run] $(ts) iter=$i" | tee -a "$LOG_FILE"
  python3 generate_llm_tables.py \
    --iter "$i" \
    --resume \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --session-id "$SESSION_ID" \
    --trial-id "$TRIAL_ID" \
    --skip-errors \
    --sleep "$SLEEP" \
    --api-url "$API_URL" \
    >>"$LOG_FILE" 2>&1
done

echo "[run] $(ts) computing similarity" | tee -a "$LOG_FILE"
python3 compute_llm_similarity_tables.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$SIMILARITY_OUT" \
  >>"$LOG_FILE" 2>&1

echo "[done] $(ts) finished" | tee -a "$LOG_FILE"
