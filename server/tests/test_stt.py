import os, io, json
from services.stt_app import app

def test_stt_smoke(tmp_path):
    client = app.test_client()
    wav_bytes = b'RIFF0000WAVEfmt '
    data = {
        'session_id': 'demo-session',
        'trial_id': '1',
        'lang': 'auto',
        'audio': (io.BytesIO(wav_bytes), 'user_1B_mic.wav')
    }
    resp = client.post('/api/v1/stt', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'asr_text_path' in j
