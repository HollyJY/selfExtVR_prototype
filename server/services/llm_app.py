from flask import Flask, request, jsonify
import os, json, yaml, time
from common.io_paths import ensure_trial_paths
from common.timeline import Timeline
from common.logging_conf import setup_logging
import requests

CFG_PATH = "config/app.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

app = Flask(__name__)
log = setup_logging("llm")

# --- Ollama client (preload model once) ---
# Normalize host: ensure scheme present
_raw_host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
if not (_raw_host.startswith('http://') or _raw_host.startswith('https://')):
    _raw_host = 'http://' + _raw_host
OLLAMA_HOST = _raw_host
# Prefer explicit user request model, fallback to config, then a sane default
MODEL_NAME = os.environ.get('OLLAMA_MODEL') or cfg.get('llm', {}).get('model_name') or 'llama3.1:8b-instruct-q8_0'

_http = requests.Session()

def _ollama_generate(prompt: str, keep_alive: str = '24h', timeout: float = 120.0) -> str:
    try:
        url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
        data = {
            'model': MODEL_NAME,
            'prompt': prompt,
            'stream': False,
            'keep_alive': keep_alive,
        }
        resp = _http.post(url, json=data, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        # Ollama returns {'response': '...'}
        return j.get('response', '').strip()
    except Exception as e:
        log.warning(f"Ollama call failed: {e}", exc_info=False)
        return ''

def _warmup_model():
    # Trigger a lightweight load to avoid first-call latency
    test_prompt = "You are loaded."
    _ = _ollama_generate(test_prompt, keep_alive='24h', timeout=10.0)
    log.info(f"Ollama warmup attempted for model '{MODEL_NAME}' at {OLLAMA_HOST}")

# Warm up on import
try:
    _warmup_model()
except Exception as _:
    # Non-fatal; service continues and will try again during requests
    pass

@app.post('/api/v1/llm')
def llm():
    payload = request.get_json()
    session_id = payload['session_id']
    trial_id = int(payload['trial_id'])
    prompt_path = payload['prompt_path']
    condition = payload.get('condition', '1')  # Optional, 1 - repeat, 2 - enhance, 3 - oppose
    if not prompt_path.startswith('data/'):
        prompt_path = os.path.join('data', prompt_path)

    paths = ensure_trial_paths(session_id, trial_id)
    # Output file name as requested
    out_path = os.path.join(paths['trial_dir'], 'user_2B_llm.txt')
    print(out_path)

    tl = Timeline(paths['timeline_path'])
    tl.add('llm_start')

    # Build prompt: template (prompt_llm.txt) + scene + ASR text
    prompt_root = cfg.get('paths', {}).get('prompt_root')
    # Fallback to local path if configured root doesn't exist
    if not prompt_root or not os.path.isdir(prompt_root):
        # Try relative to current working dir (e.g., 'server')
        candidate = os.path.join('material', 'prompts')
        if os.path.isdir(candidate):
            prompt_root = candidate
        else:
            # As a last resort, try relative to this file location
            here = os.path.dirname(os.path.dirname(__file__))  # services/
            prompt_root = os.path.join(here, 'material', 'prompts')

    prompt_tmpl_path = os.path.join(prompt_root, 'prompt_llm.txt')
    try:
        with open(prompt_tmpl_path, 'r', encoding='utf-8') as f:
            tmpl = f.read().strip()
    except Exception as e:
        tmpl = ''
        log.warning(f"Prompt template not found or unreadable at {prompt_tmpl_path}: {e}")

    # Read scene text from session meta (e.g., .../sessions/<session>_session/trial_<trial_id>/scene.txt)
    scene_path = os.path.join(paths['trial_dir'], 'scene.txt')
    try:
        with open(scene_path, 'r', encoding='utf-8') as f:
            scene_text = f.read().strip()
    except Exception as e:
        scene_text = ''
        log.warning(f"Scene not found or unreadable at {scene_path}: {e}")

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            asr_text = f.read().strip()
    except Exception as e:
        asr_text = ''
        log.warning(f"ASR prompt not found or unreadable at {prompt_path}: {e}")

    # Build: template + scene + ASR text
    combined_prompt = "\n\n".join(part for part in [tmpl, scene_text, asr_text] if part).strip()

    # Call Ollama; fallback to echoing prompt if unavailable
    reply = _ollama_generate(combined_prompt)
    if not reply:
        reply = combined_prompt if combined_prompt else 'No prompt content available.'

    # Optionally modify reply based on condition
    # If the model returned multiple numbered options (e.g., "1. Keep...", "2. Enhance...", "3. Oppose..."),
    # pick the section that corresponds to the requested condition.
    try:
        import re

        # Condition is expected to be '1', '2' or '3'
        try:
            cond_num = int(condition)
        except Exception:
            cond_num = 1
        if cond_num not in (1, 2, 3):
            cond_num = 1

        # Simple approach per request:
        # - split reply into paragraphs separated by blank lines
        # - find the paragraph that contains the condition number (standalone digit)
        # - return the paragraph that follows it (the block after the header)
        blocks = [b.strip() for b in re.split(r'\n\s*\n', reply) if b.strip()]
        picked = None
        digit_re = re.compile(rf'(?<!\d){cond_num}(?!\d)')
        for i, block in enumerate(blocks):
            if digit_re.search(block):
                # return the next block if available
                if i + 1 < len(blocks):
                    picked = blocks[i + 1]
                else:
                    # no next block; fall back to the block itself
                    picked = block
                break

        if picked:
            # Cleanup: remove trailing decorations like '**', ensure single line, and drop any header-like line.
            cleaned = picked.strip()
            # strip trailing asterisks/decorations at end
            cleaned = re.sub(r'\s*\*+\s*$', '', cleaned)
            # split to lines and remove empty
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            if len(lines) > 1:
                # remove any line that contains the standalone header number
                header_line_re = re.compile(rf'(?<!\d){cond_num}(?!\d)')
                lines = [ln for ln in lines if not header_line_re.search(ln)]
                if not lines:
                    # if everything got removed, fall back to first non-empty original line
                    lines = [cleaned.splitlines()[0].strip()]
            # join remaining lines into a single line
            one_line = ' '.join(lines) if isinstance(lines, list) else cleaned
            # remove any lingering header-like prefix at start (e.g., '**1) ' or '1. ')
            one_line = re.sub(r'^\**\s*\d+\s*[\.\)\-]\s*', '', one_line).strip()
            # final trailing decoration strip
            one_line = re.sub(r'\s*\*+\s*$', '', one_line).strip()
            reply = one_line if one_line else picked

    except Exception:
        # fast-fail: keep original reply
        pass


    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(reply)

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
