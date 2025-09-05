import sys
import os

workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.llm_app import app
from services.common.io_paths import ensure_trial_paths

def test_llm_smoke():
    client = app.test_client()

    session_id = 'demo-session'
    trial_id = 1
    paths = ensure_trial_paths(session_id, trial_id)
    prompt_path = os.path.join(paths['trial_dir'], 'user_1B_asr.txt')

    payload = {
        "session_id": session_id,
        "trial_id": trial_id,
        "prompt_path": prompt_path,
    }
    resp = client.post('/api/v1/llm', json=payload)
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'llm_text_path' in j

if __name__ == '__main__':
    test_llm_smoke()