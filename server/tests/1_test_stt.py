import sys
import os

workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.stt_app import app
from services.common.io_paths import ensure_trial_paths

def test_stt_smoke():
    client = app.test_client()
    session_id = 'demo-session'
    trial_id = 1

    paths = ensure_trial_paths(session_id, trial_id)
    mic_audio_path = os.path.join(paths['trial_dir'], 'user_1B_mic.wav')

    data = {
        'session_id': session_id,
        'trial_id': trial_id,
        'lang': 'en',
        'audio': open(mic_audio_path, 'rb') # testing a real audio file
    }
    resp = client.post('/api/v1/stt', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200  # check HTTP response status code
    j = resp.get_json()
    assert 'asr_text_path' in j  # check if the response contains the expected key

if __name__ == '__main__':
    test_stt_smoke()