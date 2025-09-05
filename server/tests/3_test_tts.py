import sys
import os

workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.tts_app import app
from services.common.io_paths import ensure_trial_paths

def test_tts_smoke():
    client = app.test_client()
    session_id = 'demo-session'
    trial_id = 1
    paths = ensure_trial_paths(session_id, trial_id)
    ref_audio_path = os.path.abspath(os.path.join(paths['meta_dir'], 'sample_voice.wav'))
    text_path = os.path.abspath(os.path.join(paths['trial_dir'], 'user_2B_llm.txt'))

    payload = {
        "session_id": session_id,
        "trial_id": trial_id,
        # Provide explicit reference audio and text file (absolute paths)
        "ref_path": ref_audio_path,
        "text_path": text_path
    }
    resp = client.post('/api/v1/tts', json=payload)
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'audio_path' in j
    # Optional: ensure filename follows rule: replace 'llm' -> 'tts.wav'
    assert j['audio_path'].endswith('user_2B_tts.wav')

if __name__ == '__main__':
    test_tts_smoke()
