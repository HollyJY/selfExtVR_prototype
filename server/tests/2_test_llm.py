import sys

workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.llm_app import app

def test_llm_smoke():
    client = app.test_client()
    payload = {
        "session_id": "demo-session",
        "trial_id": 1,
        "prompt_path": "sessions/demo-session_session/trial_001/user_1B_asr.txt",
    }
    resp = client.post('/api/v1/llm', json=payload)
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'llm_text_path' in j

if __name__ == '__main__':
    test_llm_smoke()