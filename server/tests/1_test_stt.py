import io
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

# if __name__ == '__main__':
#     import os, requests

#     # start flask app in a separate process
#     p = Process(target=run_flask_app)
#     p.start()

#     # wait for server to run
#     import time
#     time.sleep(2)

#     SERVER = "http://127.0.0.1:7001"  # or internal IP
#     INPUT_AUDIO = "tests/test_data/0_sample_audio/sample_zjy.mp3"

#     resp = requests.post(
#         f"{SERVER}/api/v1/stt",
#         files={"audio": open(INPUT_AUDIO, "rb")},
#         data={"session_id": "demo-session", "trial_id": "000", "lang": "en"}
#     )
#     print(resp.text)  # 打印原始响应内容
#     print(resp.json())

#     p.terminate()