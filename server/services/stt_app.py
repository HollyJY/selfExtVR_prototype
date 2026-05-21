from flask import Flask, request, jsonify, send_from_directory
import os, time, hashlib, yaml
from common.io_paths import ensure_trial_paths
from common.timeline import Timeline
from common.logging_conf import setup_logging

import whisper

# load config
CFG_PATH = "config/app.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

MODEL_NAME = os.environ.get("WHISPER_MODEL", cfg.get("stt", {}).get("model_size", "base.en"))
REQUESTED_DEVICE = os.environ.get("WHISPER_DEVICE", cfg.get("stt", {}).get("device", "cpu")) # cpu or cuda
MODELS_DIR = cfg["paths"]["models_root"] if "paths" in cfg else "models"
STT_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")


def _load_model_with_fallback():
    requested = (REQUESTED_DEVICE or "cpu").strip().lower()
    print(f"[stt] loading whisper model='{MODEL_NAME}' on device='{requested}'...")
    try:
        model = whisper.load_model(MODEL_NAME, download_root=STT_MODELS_DIR).to(requested)
        print(f"[stt] model ready on device='{requested}'")
        return model, requested
    except Exception as e:
        if requested == "cpu":
            raise
        fallback = "cpu"
        print(f"[stt] failed to initialize on device='{requested}': {e}")
        print(f"[stt] retrying whisper model='{MODEL_NAME}' on device='{fallback}'...")
        model = whisper.load_model(MODEL_NAME, download_root=STT_MODELS_DIR).to(fallback)
        print(f"[stt] model ready on device='{fallback}'")
        return model, fallback


_model, ACTIVE_DEVICE = _load_model_with_fallback()

# create the Flask app
app = Flask(__name__)
log = setup_logging("stt")

# constants
DATA_ROOT = cfg["paths"]["data_root"] if "paths" in cfg else "data"

@app.post('/api/v1/stt')
def stt():
    """
    Form-Data:
      audio: binary audio file, mp3 or wav
      session_id: string
      trial_id: integer
      lang: for example 'en' or 'auto'
    """
    try:
        # parse form data
        if "audio" not in request.files:
            return jsonify({"error": "missing file field 'audio'"}), 400

        session_id = request.form.get("session_id", "demo-session")
        trial_id = int(request.form.get("trial_id", "000"))
        audio = request.files['audio']

        lang = request.form.get('lang','en')

        # save the audio file
        paths = ensure_trial_paths(session_id, trial_id)
        wav_path = os.path.join(paths['trial_dir'], 'user_1B_mic.wav')
        audio.save(wav_path)

        # log the request
        tl = Timeline(paths['timeline_path'])
        tl.add('recv_start', lang=lang)

        # TODO load Whisper and transcribe the wav_path
        # Whisper's language parameter uses a two-letter code. 
        transcribe_kwargs = dict(fp16=False) if ACTIVE_DEVICE == "cpu" else dict(fp16=True)
        if lang != "auto":
            transcribe_kwargs["language"] = lang

        result = _model.transcribe(wav_path, **transcribe_kwargs)
        text = result.get("text", "").strip()

        asr_text_path = os.path.join(paths['trial_dir'], 'user_1B_asr.txt')
        with open(asr_text_path, 'w', encoding='utf-8') as f:
            f.write(text + "\n")

        tl.add('asr_end')

        return jsonify({
            'session_id': session_id,
            'trial_id': trial_id,
            'asr_text_path': os.path.relpath(asr_text_path, start='data'),
            'asr_confidence': 0.90,
            'duration_sec': 0.0,
            'timeline': tl.snapshot(),
        })
    except Exception as e:
        import traceback
        log.error(f"Error processing request: {traceback.format_exc()}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.get('/healthz')
def healthz():
    return jsonify({"service":"stt","ts":time.time()})

@app.get('/files/<path:p>')
def fileserve(p):
    return send_from_directory('data', p, as_attachment=True)

if __name__ == '__main__':
    port = cfg.get("http", {}).get("stt_port", 7001)
    app.run(host='0.0.0.0', port=port)
