# HCI Audio LLM Pipeline Skeleton

This repository contains a minimal yet production-minded skeleton for an HCI experiment pipeline that runs three Flask services inside a single Docker container.
Unity runs locally on the laptop. The container runs on a local GPU server in the same building network.

## Services
1. STT on port 7001 handles audio uploads and Whisper transcription.
2. LLM on port 7002 runs local LLM inference.
3. TTS on port 7003 synthesizes audio with IndexTTS.

The services do not call each other directly. Unity orchestrates the sequence, which keeps the dependency boundaries clean.

## Directory layout
```text
project/
  Dockerfile
  requirements.txt
  supervisor.conf
  config/
    app.yml
    voices.yml
  services/
    stt_app.py
    llm_app.py
    tts_app.py
    common/
      io_paths.py
      gpu_lock.py
      timeline.py
      logging_conf.py
  models/
    llama/
    whisper/
    indextts/
  data/
    sessions/
    exports/
  logs/
  tests/
    test_stt.py
    test_llm.py
    test_tts.py
```

## Build
```bash
docker build -t extselfvr:ollama server
```

## Run
run in bash, testing
```bash
docker run -it --rm --gpus all -p 7001 -p 7002 -p 7003 -p 11434 -v "$(pwd)/server:/workspace" extselfvr:ollama bash
```

## Endpoints
- STT: `POST /api/v1/stt` accepts `multipart/form-data` with `audio`, `session_id`, `trial_id`, optional `lang`.
- LLM: `POST /api/v1/llm` accepts JSON with `session_id`, `trial_id`, and `prompt_path` pointing to a file under `data/`.
- TTS: `POST /api/v1/tts` accepts JSON with `session_id`, `trial_id`.

Each service provides `GET /healthz` and file serving via `GET /files/<path>` where applicable.

## Notes
- Replace the TODO blocks with your actual Whisper, Llama, and IndexTTS calls.
- The GPU lock helper is available if you place any service on `cuda` in a single-GPU environment.
- All code is in English for team collaboration.

---

## run the service in the container + call the service from outside the container

1. open container

```bash
docker run -it --rm --gpus all \
  -p 7001:7001 -p 7002:7002 -p 7003:7003 -p 11434:11434 \
  -v "$(pwd)/server:/workspace" \
  extselfvr:ollama bash
```

2. run the service
```bash
python3 /workspace/services/stt_app.py &
python3 /workspace/services/llm_app.py &
python3 /workspace/services/tts_app.py &
```

3. test pipeline
```bash
cd /home/holly/extSelfVR_prototype/server
python3 tests/test_outside.py
```
* test healthiness
```bash
curl http://localhost:7001/healthz

curl http://localhost:7002/healthz

curl http://localhost:7003/healthz
```