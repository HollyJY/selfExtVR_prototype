from flask import Flask, request, jsonify, send_from_directory
import os, yaml, time, traceback
from typing import Optional
from services.common.io_paths import ensure_trial_paths
from services.common.timeline import Timeline
from services.common.logging_conf import setup_logging

CFG_PATH = "config/app.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

app = Flask(__name__)
log = setup_logging("tts")

# Ensure local IndexTTS package is importable when running from repo
try:
    from indextts.infer import IndexTTS  # type: ignore
except Exception:  # pragma: no cover - allow dynamic path fix
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    local_indextts = repo_root / "server" / "models" / "indexTTS"
    if local_indextts.exists():
        sys.path.insert(0, str(local_indextts))
        log.info(f"[tts] Added local IndexTTS path: {local_indextts}")
        from indextts.infer import IndexTTS  # type: ignore
    else:
        raise

# Discover a viable model directory (contains tts.yml)
def _find_model_dir() -> Optional[str]:
    candidates = []
    env_dir = os.environ.get("INDEXTTS_MODEL_DIR")
    if env_dir:
        candidates.append(env_dir)

    # Config models_root + indextts/checkpoints
    try:
        models_root = cfg.get("paths", {}).get("models_root")
        if models_root:
            candidates.append(os.path.join(models_root, "indextts", "checkpoints"))
    except Exception:
        pass

    # Repo-local fallback
    candidates.extend([
        os.path.join("server", "models", "indexTTS", "checkpoints"),
        os.path.join("models", "indexTTS", "checkpoints"),
        "checkpoints",
    ])

    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "config.yaml")):
            return d
    return None

# Load IndexTTS model once (first import), similar to STT service
log.info("[tts] Initializing IndexTTS model (one-time load)...")
_MODEL_DIR = _find_model_dir()
if not _MODEL_DIR:
    log.warning("[tts] Could not locate IndexTTS checkpoints (config.yaml not found). Inference will fail.")
    _tts_model = None
else:
    try:
        _tts_model = IndexTTS(model_dir=_MODEL_DIR, cfg_path=os.path.join(_MODEL_DIR, "config.yaml"))
        log.info(f"[tts] IndexTTS ready. model_dir='{_MODEL_DIR}'")
    except Exception:
        log.error("[tts] Failed to initialize IndexTTS:\n" + traceback.format_exc())
        _tts_model = None

def _resolve_text(paths: dict, payload: dict) -> str:
    # Priority: explicit text in request -> text_path file -> LLM output file -> fallback
    txt = (payload or {}).get("text")
    if txt and isinstance(txt, str) and txt.strip():
        return txt.strip()

    text_path = (payload or {}).get("text_path")
    if text_path:
        candidates = [text_path]
        if not os.path.isabs(text_path):
            candidates.append(os.path.join('data', text_path))
            candidates.append(os.path.join(paths['trial_dir'], os.path.basename(text_path)))
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        txt = f.read().strip()
                        if txt:
                            return txt
                except Exception:
                    pass

    llm_path = os.path.join(paths['trial_dir'], 'llama_output.txt')
    if os.path.isfile(llm_path):
        try:
            with open(llm_path, 'r', encoding='utf-8') as f:
                txt = f.read().strip()
                if txt:
                    return txt
        except Exception:
            pass
    return "Hello, this is a test."

def _resolve_ref_path(paths: dict, payload: dict) -> str:
    # Allow client to pass a ref path; else fall back to bundled sample audio
    # should be under current session folder/meta/
    ref = (payload or {}).get("ref_path")
    if ref:
        # If not already under data, accept as given as absolute or relative
        if ref.startswith('data' + os.sep):
            return ref
        return os.path.join(paths['base'], ref)
    # Fallback to a known test audio as reference timbre
    default_ref = os.path.join('tests', 'test_data', '0_sample_audio', 'sample_zjy.wav')
    return default_ref

def _derive_output_name(paths: dict, payload: dict) -> str:
    """Derive output wav filename from input text file name.
    If payload has text_path like 'user_2B_llm.txt', output becomes 'user_2B_tts.wav'.
    Fallback: 'npc_2A_tts.wav'.
    """
    text_path = (payload or {}).get("text_path")
    if text_path:
        base = os.path.basename(text_path)
        name, _ext = os.path.splitext(base)
        if 'llm' in name:
            out_name = name.replace('llm', 'tts') + '.wav'
        else:
            out_name = f"{name}_tts.wav"
        return os.path.join(paths['trial_dir'], out_name)
    return os.path.join(paths['trial_dir'], 'npc_2A_tts.wav')

@app.post('/api/v1/tts')
def tts():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get('session_id', 'demo-session')
        trial_id = int(payload.get('trial_id', 0))

        paths = ensure_trial_paths(session_id, trial_id)
        audio_path = _derive_output_name(paths, payload)

        tl = Timeline(paths['timeline_path'])
        text = _resolve_text(paths, payload)
        ref_path = _resolve_ref_path(paths, payload)
        tl.add('tts_start', text_len=len(text), ref=os.path.basename(ref_path))

        did_fallback = False
        try:
            if _tts_model is None:
                raise RuntimeError("IndexTTS model is not initialized")
            # Perform inference (batch-friendly settings from sample)
            _ = _tts_model.infer_fast(
                ref_path,
                text,
                audio_path,
                verbose=bool(payload.get('verbose', False)),
                max_text_tokens_per_sentence=120,
                sentences_bucket_max_size=4,
                do_sample=True,
                top_p=0.8,
                top_k=30,
                temperature=1.0,
                length_penalty=0.0,
                num_beams=3,
                repetition_penalty=10.0,
                max_mel_tokens=600,
            )
        except Exception:
            # Fallback: generate a short silent wav to keep pipeline flowing
            log.error("[tts] Inference failed, writing fallback WAV:\n" + traceback.format_exc())
            try:
                import wave, struct
                sr = int(cfg.get('tts', {}).get('sample_rate', 48000))
                n_channels = 1
                sampwidth = 2  # 16-bit
                duration_sec = 0.2
                n_frames = int(sr * duration_sec)
                with wave.open(audio_path, 'wb') as wf:
                    wf.setnchannels(n_channels)
                    wf.setsampwidth(sampwidth)
                    wf.setframerate(sr)
                    silence_frame = struct.pack('<h', 0)
                    wf.writeframes(silence_frame * n_frames)
                did_fallback = True
                tl.add('tts_fallback')
            except Exception:
                raise

        tl.add('tts_end')

        return jsonify({
            'session_id': session_id,
            'trial_id': trial_id,
            'audio_path': os.path.relpath(audio_path, start='data'),
            'timeline': tl.snapshot(),
            'fallback': did_fallback
        })
    except Exception as e:
        log.error(f"[tts] Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.get('/healthz')
def healthz():
    return jsonify({"service":"tts","ts":time.time()})

@app.get('/files/<path:p>')
def fileserve(p):
    return send_from_directory('data', p, as_attachment=True)

if __name__ == '__main__':
    port = cfg.get("http", {}).get("tts_port", 7003)
    app.run(host='0.0.0.0', port=port)
