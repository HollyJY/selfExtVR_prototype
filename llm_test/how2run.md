### generate answer

```bash
caffeinate -dimsu zsh -lc '
for i in $(seq 1 200); do
  echo "[run] iter=$i"
  python3 generate_llm_tables.py \
    --iter "$i" \
    --resume \
    --input final_12_scenes.csv \
    --output-dir llm_tables_out \
    --session-id "test_llm" \
    --trial-id 1 \
    --skip-errors \
    --sleep 0.3 || exit 1
done
'
```

### Run with Docker (Windows / macOS / Linux)

1) Build the image (run at repo root)  
```bash
docker build -t llm-test -f llm_test/Dockerfile .
# Apple Silicon needing x86 compatibility:
# docker build --platform linux/amd64 -t llm-test -f llm_test/Dockerfile .
```

2) Run a container and mount local `llm_test` (outputs written back to host)
- macOS / Linux:
```bash
docker run --rm -it -v "$(pwd)/llm_test:/app/llm_test" \
  -e LLM_API_URL=http://192.168.37.177:7002/api/v1/llm \
  llm-test
```
- Windows PowerShell:
```powershell
docker run --rm -it -v ${PWD}/llm_test:/app/llm_test `
  -e LLM_API_URL=http://192.168.37.177:7002/api/v1/llm `
  llm-test
```
- Windows CMD:
```cmd
docker run --rm -it -v %cd%\llm_test:/app/llm_test ^
  -e LLM_API_URL=http://192.168.37.177:7002/api/v1/llm ^
  llm-test
```

3) Inside the container, run scripts (outputs land in host `llm_test`)
- Generate answer tables:
```bash
python generate_llm_tables.py --iter 1 --resume --input final_12_scenes.csv \
  --output-dir llm_tables_out --session-id test_llm --trial-id 1 \
  --skip-errors --sleep 0.3
```
- Compute similarity tables (auto-selects CUDA/MPS/CPU):
```bash
python compute_llm_similarity_tables.py --input-dir llm_tables_out \
  --output-dir llm_tables_similarity_out --device auto --hf-device auto
```

Tip: If the backend API URL differs, override with env `LLM_API_URL`. `prompt_path` is auto-normalized to POSIX in code, so no manual backslash fixes are needed across platforms.

### Keep jobs running after you disconnect / close the lid

- Recommended: start a server-side session (tmux/screen) and run the commands inside it. Example:
```bash
tmux new -s llmjob
# inside tmux
bash llm_test/run_headless.sh
# detach with Ctrl-b d; reattach with tmux attach -t llmjob
```

- If tmux is unavailable, launch a nohup background job (survives SSH drop):
```bash
cd /home/holly/extSelfVR_prototype
nohup bash llm_test/run_headless.sh > llm_test/run_nohup.log 2>&1 &
echo $!  # prints PID; you can kill it with `kill <pid>`
```

`run_headless.sh` already uses `--resume`, so you can rerun it later and it will pick up existing CSV progress. Adjust iterations via env vars, e.g. `ITER_END=50`, `ITER_START=10`, or change API with `LLM_API_URL=...`.
