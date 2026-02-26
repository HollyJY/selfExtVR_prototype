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

WORKSPACE_ROOT = os.environ.get('WORKSPACE', '/workspace')
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _resolve_user_context(raw_value, paths):
    """Return (text, source_path) for the provided user_context value."""
    value = (raw_value or '').strip()
    if not value:
        return '', ''

    candidates = []

    def _add_candidate(p: str):
        if p:
            candidates.append(p)

    if os.path.isabs(value):
        _add_candidate(value)
    else:
        _add_candidate(value)
        _add_candidate(os.path.join(paths['trial_dir'], value))
        _add_candidate(os.path.join(paths['meta_dir'], value))
        _add_candidate(os.path.join(paths['base'], value))
        _add_candidate(os.path.join('data', value))
        if WORKSPACE_ROOT:
            _add_candidate(os.path.join(WORKSPACE_ROOT, value.lstrip('/')))
            _add_candidate(os.path.join(WORKSPACE_ROOT, 'data', value))
        _add_candidate(os.path.join(REPO_ROOT, value))

    seen = set()
    for cand in candidates:
        cand_abs = os.path.abspath(cand)
        if cand_abs in seen:
            continue
        seen.add(cand_abs)
        if os.path.isfile(cand_abs):
            try:
                with open(cand_abs, 'r', encoding='utf-8') as f:
                    return f.read().strip(), cand_abs
            except Exception as e:
                log.warning(f"User context file not readable at {cand_abs}: {e}")
                return '', cand_abs

    return value, ''

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
    condition = str(payload.get('condition', '')).strip()  # Optional, 1 - repeat, 2 - enhance, 3 - oppose
    cond_num = None
    try:
        parsed = int(condition)
        if parsed in (1, 2, 3, -1):
            cond_num = parsed
    except Exception:
        cond_num = None
    user_context_raw = payload.get('user_context', '')
    if not prompt_path.startswith('data/'):
        prompt_path = os.path.join('data', prompt_path)

    paths = ensure_trial_paths(session_id, trial_id)
    # Output file name as requested
    out_path = os.path.join(paths['trial_dir'], 'user_2B_llm.txt')
    print(out_path)

    tl = Timeline(paths['timeline_path'])
    tl.add('llm_start', user_context=bool(str(user_context_raw).strip()))

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

    tmpl = ''
    prompt_tmpl_path = None
    if cond_num == -1:
        log.info("Condition -1 received: skipping prompt template load")
    else:
        prompt_tmpl_path = os.path.join(prompt_root, 'prompt_llm.txt')
        if cond_num in (1, 2, 3):
            prompt_tmpl_name = f'prompt_llm_cond{cond_num}.txt'
            prompt_tmpl_path_candidate = os.path.join(prompt_root, prompt_tmpl_name)
            if os.path.isfile(prompt_tmpl_path_candidate):
                prompt_tmpl_path = prompt_tmpl_path_candidate
        try:
            with open(prompt_tmpl_path, 'r', encoding='utf-8') as f:
                tmpl = f.read().strip()
        except Exception as e:
            tmpl = ''
            log.warning(f"Prompt template not found or unreadable at {prompt_tmpl_path}: {e}")

    # Resolve user-provided context first; fallback to the on-disk scene file
    scene_text = ''
    context_source = ''
    if user_context_raw:
        scene_text, context_source = _resolve_user_context(user_context_raw, paths)
        if scene_text and not context_source:
            context_source = 'inline_user_context'
        if not scene_text:
            log.warning("User context provided but empty after processing. Falling back to scene.txt")
    if not scene_text:
        scene_path = os.path.join(paths['trial_dir'], 'scene.txt')
        try:
            with open(scene_path, 'r', encoding='utf-8') as f:
                scene_text = f.read().strip()
                context_source = scene_path if scene_text else ''
        except Exception as e:
            scene_text = ''
            context_source = ''
            log.warning(f"Scene not found or unreadable at {scene_path}: {e}")

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            asr_text = f.read().strip()
    except Exception as e:
        asr_text = ''
        log.warning(f"ASR prompt not found or unreadable at {prompt_path}: {e}")

    # Build: template + scene + ASR text
    tl.add('llm_context', context_source=context_source or ('scene_file' if scene_text else 'none'))

    combined_prompt = "\n\n".join(part for part in [tmpl, scene_text, asr_text] if part).strip()

    # Call Ollama; fallback to echoing prompt if unavailable
    reply = _ollama_generate(combined_prompt)
    if not reply:
        reply = combined_prompt if combined_prompt else 'No prompt content available.'

    # Final cleanup for output format consistency
    try:
        import re
        cleaned = (reply or '').strip()
        cleaned = re.sub(r'\*+', '', cleaned)
        cleaned = re.sub(r'(?i)\bargument\b[:\s-]*', '', cleaned)
        cleaned = re.sub(r"(?i)here'?s\s+the\s+response\s+in\s+2-3\s+sentences\s*[:.,-]*", '', cleaned)
        cleaned = re.sub(r"(?i)response\s+in\s+2-3\s+sentences\s*[:.,-]*", '', cleaned)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'^\s*\d+\s*[\.\)\-]\s*', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            reply = cleaned

    except Exception:
        # fast-fail: keep original reply
        pass


    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(reply)

    tl.add('llm_end')

    return jsonify({
        'llm_text_path': os.path.relpath(out_path, start='data'),
        'llm_text': reply,
        'timeline': tl.snapshot()
    })

@app.get('/healthz')
def healthz():
    return jsonify({"service":"llm","ts":time.time()})

if __name__ == '__main__':
    port = cfg.get("http", {}).get("llm_port", 7002)
    app.run(host='0.0.0.0', port=port)
