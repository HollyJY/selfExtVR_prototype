from flask import Flask, request, jsonify
import os, json, yaml, time
from services.common.io_paths import ensure_trial_paths
from services.common.timeline import Timeline
from services.common.logging_conf import setup_logging

CFG_PATH = "config/app.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

app = Flask(__name__)
log = setup_logging("llm")

@app.post('/api/v1/llm')
def llm():
    payload = request.get_json()
    session_id = payload['session_id']
    trial_id = int(payload['trial_id'])
    prompt_path = payload['prompt_path']
    if not prompt_path.startswith('data/'):
        prompt_path = os.path.join('data', prompt_path)

    paths = ensure_trial_paths(session_id, trial_id)
    out_path = os.path.join(paths['trial_dir'], 'llama_output.txt')

    tl = Timeline(paths['timeline_path'])
    tl.add('llm_start')

    # TODO call your local LLM and write the output to out_path
    open(out_path, 'w', encoding='utf-8').write('stub reply')

    tl.add('llm_end')

    return jsonify({
        'llm_text_path': os.path.relpath(out_path, start='data'),
        'timeline': tl.snapshot()
    })

@app.get('/healthz')
def healthz():
    return jsonify({"service":"llm","ts":time.time()})

if __name__ == '__main__':
    port = cfg.get("http", {}).get("llm_port", 7002)
    app.run(host='0.0.0.0', port=port)
