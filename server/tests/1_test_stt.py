import os, io, json
# from services.stt_app import app

# easy test for whisper model
import whisper

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

if __name__ == '__main__':
    import time

    # simple test of whisper model
    INPUT_FOLDER_STT = 'server/tests/test_data/0_sample_audio'
    OUTPUT_FOLDER_STT = 'server/tests/test_data/1_output_stt'
    input_audio_path = os.path.join(INPUT_FOLDER_STT, 'sample_zjy.mp3')

    start_time = time.time()

    # download the model to server/models/whisper
    model = whisper.load_model("base.en", download_root="./server/models/whisper")
    
    # run the model
    result = model.transcribe(input_audio_path, language='en', fp16=False)

    # save the result
    with open(os.path.join(OUTPUT_FOLDER_STT, 'sample_zjy_whisper.txt'), 'w') as f:
        f.write(result['text'] + '\n')
    print(result['text'], '\n')
    print(f"whisper STT done in {time.time() - start_time} seconds\n\n")