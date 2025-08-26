from flask import Flask, request, jsonify, send_from_directory
import os, yaml, time
from services.common.io_paths import ensure_trial_paths
from services.common.timeline import Timeline
from services.common.logging_conf import setup_logging

CFG_PATH = "config/app.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

app = Flask(__name__)
log = setup_logging("tts")

@app.post('/api/v1/tts')
def tts():
    payload = request.get_json()
    session_id = payload['session_id']
    trial_id = int(payload['trial_id'])

    paths = ensure_trial_paths(session_id, trial_id)
    audio_path = os.path.join(paths['trial_dir'], 'npc_2A_tts.wav')

    tl = Timeline(paths['timeline_path'])
    tl.add('tts_start')

    # TODO call IndexTTS to synthesize audio and write to audio_path
    # For now write a tiny placeholder WAV header that will not be a valid audio
    with open(audio_path,'wb') as f:
        f.write(b'RIFF0000WAVEfmt ')

    tl.add('tts_end')

    return jsonify({
        'audio_path': os.path.relpath(audio_path, start='data'),
        'timeline': tl.snapshot()
    })

@app.get('/healthz')
def healthz():
    return jsonify({"service":"tts","ts":time.time()})

@app.get('/files/<path:p>')
def fileserve(p):
    return send_from_directory('data', p, as_attachment=True)

if __name__ == '__main__':
    port = cfg.get("http", {}).get("tts_port", 7003)
    app.run(host='0.0.0.0', port=port)
