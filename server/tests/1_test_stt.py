import sys

workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.stt_app import app

def test_stt_smoke():
    client = app.test_client()
    # wav_bytes = b'RIFF0000WAVEfmt '  # simulate a simple WAV file header
    data = {
        'session_id': 'demo-session',
        'trial_id': '1',
        'lang': 'en',
        'audio': open('/workspace/tests/test_data/0_sample_audio/sample_zjy.wav', 'rb') # testing a real audio file
    }
    resp = client.post('/api/v1/stt', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200  # check HTTP response status code
    j = resp.get_json()
    assert 'asr_text_path' in j  # check if the response contains the expected key

if __name__ == '__main__':
    test_stt_smoke()