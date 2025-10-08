import sys, time, os

workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

# Import apps with distinct names
from services import stt_app as stt_app_module
from services import llm_app as llm_app_module
from services import tts_app as tts_app_module
from services.common.io_paths import ensure_trial_paths


def run_pipeline(session_id: str = 'demo-session', trial_id: int = 1):
    """Run STT -> LLM -> TTS sequentially using Flask test clients.
    Returns (durations_dict).
    """
    stt_client = stt_app_module.app.test_client()
    llm_client = llm_app_module.app.test_client()
    tts_client = tts_app_module.app.test_client()

    timings = {}
    t0 = time.perf_counter()

    # 1) STT
    stt_start = time.perf_counter()
    # Build trial paths to locate the sample voice under the trial folder meta
    trial_paths = ensure_trial_paths(session_id, trial_id)
    stt_data = {
        'session_id': session_id,
        'trial_id': str(trial_id),
        'lang': 'en',
        'audio': open(os.path.abspath(os.path.join(trial_paths['trial_dir'], 'user_1B_mic.wav')), 'rb')
    }
    stt_resp = stt_client.post('/api/v1/stt', data=stt_data, content_type='multipart/form-data')
    assert stt_resp.status_code == 200, f"STT HTTP {stt_resp.status_code}: {stt_resp.data!r}"
    stt_json = stt_resp.get_json()
    assert 'asr_text_path' in stt_json, f"STT missing asr_text_path: {stt_json}"
    asr_rel_path = stt_json['asr_text_path']
    timings['stt_sec'] = time.perf_counter() - stt_start

    # 2) LLM
    llm_start = time.perf_counter()
    llm_payload = {
        'session_id': session_id,
        'trial_id': trial_id,
        'prompt_path': asr_rel_path,  # relative to data/
    }
    llm_resp = llm_client.post('/api/v1/llm', json=llm_payload)
    assert llm_resp.status_code == 200, f"LLM HTTP {llm_resp.status_code}: {llm_resp.data!r}"
    llm_json = llm_resp.get_json()
    assert 'llm_text_path' in llm_json, f"LLM missing llm_text_path: {llm_json}"
    llm_rel_path = llm_json['llm_text_path']
    timings['llm_sec'] = time.perf_counter() - llm_start

    # 3) TTS
    tts_start = time.perf_counter()
    # Build absolute paths for TTS inputs
    # Absolute text path from the LLM result (returned relative to data/)
    text_abs_path = os.path.abspath(os.path.join('data', llm_rel_path))
    # Absolute ref audio under the trial meta directory
    ref_abs_path = os.path.abspath(os.path.join(trial_paths['meta_dir'], 'sample_voice.wav'))
    tts_payload = {
        'session_id': session_id,
        'trial_id': trial_id,
        'ref_path': ref_abs_path,
        'text_path': text_abs_path,
    }
    tts_resp = tts_client.post('/api/v1/tts', json=tts_payload)
    assert tts_resp.status_code == 200, f"TTS HTTP {tts_resp.status_code}: {tts_resp.data!r}"
    tts_json = tts_resp.get_json()
    assert 'audio_path' in tts_json, f"TTS missing audio_path: {tts_json}"
    timings['tts_sec'] = time.perf_counter() - tts_start

    timings['total_sec'] = time.perf_counter() - t0
    return timings


def main():
    # Optionally, ensure no external servers are running; here we use Flask test clients,
    # so everything runs sequentially in-process ("挂起所有服务"的要求在此天然满足)。
    t = run_pipeline(session_id = 'demo-session', trial_id=1)
    print(f"STT: {t['stt_sec']:.3f}s | LLM: {t['llm_sec']:.3f}s | TTS: {t['tts_sec']:.3f}s | TOTAL: {t['total_sec']:.3f}s")


if __name__ == '__main__':
    main()
