from flask import Flask, request, jsonify, send_from_directory
import os, time, hashlib, yaml
from services.common.io_paths import ensure_trial_paths
from services.common.timeline import Timeline
from services.common.logging_conf import setup_logging

CFG_PATH = "config/app.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

app = Flask(__name__)
log = setup_logging("stt")

DATA_ROOT = cfg["paths"]["data_root"] if "paths" in cfg else "data"

@app.post('/api/v1/stt')
def stt():
    session_id = request.form['session_id']
    trial_id = int(request.form['trial_id'])
    audio = request.files['audio']
    lang = request.form.get('lang','auto')

    paths = ensure_trial_paths(session_id, trial_id)
    wav_path = os.path.join(paths['trial_dir'], 'user_1B_mic.wav')
    audio.save(wav_path)

    tl = Timeline(paths['timeline_path'])
    tl.add('recv_start', lang=lang)

    # TODO load Whisper and transcribe the wav_path
    asr_text_path = os.path.join(paths['trial_dir'], 'user_1B_asr.txt')
    with open(asr_text_path, 'w', encoding='utf-8') as f:
        f.write('hello from stub')

    tl.add('asr_end')

    return jsonify({
        'session_id': session_id,
        'trial_id': trial_id,
        'asr_text_path': os.path.relpath(asr_text_path, start='data'),
        'asr_confidence': 0.90,
        'duration_sec': 0.0,
        'timeline': tl.snapshot(),
    })

@app.get('/healthz')
def healthz():
    return jsonify({"service":"stt","ts":time.time()})

@app.get('/files/<path:p>')
def fileserve(p):
    return send_from_directory('data', p, as_attachment=True)

if __name__ == '__main__':
    port = cfg.get("http", {}).get("stt_port", 7001)
    app.run(host='0.0.0.0', port=port)
