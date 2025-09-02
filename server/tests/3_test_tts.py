import sys
workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.tts_app import app

def test_tts_smoke():
    client = app.test_client()
    payload = {
        "session_id": "demo-session",
        "trial_id": 1,
        # Provide explicit reference audio and text file
        "ref_path": "sample_demo-session.wav",
        "text_path": "user_2B_llm.txt"
    }
    resp = client.post('/api/v1/tts', json=payload)
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'audio_path' in j
    # Optional: ensure filename follows rule: replace 'llm' -> 'tts.wav'
    assert j['audio_path'].endswith('user_2B_tts.wav')

if __name__ == '__main__':
    test_tts_smoke()
