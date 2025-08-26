from services.tts_app import app

def test_tts_smoke():
    client = app.test_client()
    payload = {
        "session_id": "demo-session",
        "trial_id": 1
    }
    resp = client.post('/api/v1/tts', json=payload)
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'audio_path' in j
