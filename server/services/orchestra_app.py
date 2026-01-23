"""
Orchestra App - Main orchestrator for STT -> LLM -> TTS pipeline
This service coordinates the full audio processing pipeline for Unity client.
"""

import os
import sys
import requests
import json
import time
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging

# Add workspace to path for imports
workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from services.common.logging_conf import setup_logging
from services.common.io_paths import ensure_trial_paths
from services.common.timeline import Timeline

# Initialize Flask app
app = Flask(__name__)

# Setup logging - remove the level parameter
log = setup_logging("orchestra")

# Service URLs (internal container communication)
STT_URL = "http://localhost:7001"
LLM_URL = "http://localhost:7002"  
TTS_URL = "http://localhost:7003"

# Configuration
UPLOAD_FOLDER = '/workspace/data/uploads'
OUTPUT_FOLDER = '/workspace/data/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SERVER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_VOICE_ID = 'robotic'
DEFAULT_REF_CANDIDATES = [
    os.path.join('tests', 'test_data', '0_sample_audio', 'neutral_sample.wav'),
    os.path.join('tests', 'test_data', 'neutral_sample.wav'),
]

CALL_LOG_FILENAME = 'call_log.jsonl'


def _resolve_audio_path(path_value, paths=None):
    """Resolve an audio file path across common repo/workspace locations."""
    if not path_value:
        return ''

    path_value = path_value.strip()
    candidates = []
    if os.path.isabs(path_value):
        candidates.append(path_value)
    else:
        candidates.append(path_value)
        if paths:
            candidates.append(os.path.join(paths['trial_dir'], path_value))
            candidates.append(os.path.join(paths['base'], path_value))
        if not path_value.startswith('data' + os.sep):
            candidates.append(os.path.join('data', path_value))
        if workspace_path:
            candidates.append(os.path.join(workspace_path, path_value.lstrip('/')))
            if not path_value.startswith('data' + os.sep):
                candidates.append(os.path.join(workspace_path, 'data', path_value))
        candidates.append(os.path.join(SERVER_ROOT, path_value))
        repo_root = os.path.abspath(os.path.join(SERVER_ROOT, '..'))
        candidates.append(os.path.join(repo_root, path_value))

    seen = set()
    for cand in candidates:
        if not cand:
            continue
        cand_abs = os.path.abspath(cand)
        if cand_abs in seen:
            continue
        seen.add(cand_abs)
        if os.path.isfile(cand_abs):
            return cand_abs
    return ''


def _get_default_ref_path(paths):
    for candidate in DEFAULT_REF_CANDIDATES:
        resolved = _resolve_audio_path(candidate, paths)
        if resolved:
            return resolved
    log.warning(
        "Default reference sample not found at expected locations. Falling back to %s",
        DEFAULT_REF_CANDIDATES[0]
    )
    return DEFAULT_REF_CANDIDATES[0]


def _determine_voice_and_ref(raw_voice_id, raw_ref_path, session_ref_path, paths):
    """Apply request voice preferences while ensuring we point at a real file."""
    requested_voice = (raw_voice_id or '').strip()
    requested_ref = (raw_ref_path or '').strip()
    default_ref = _get_default_ref_path(paths)
    session_ref = _resolve_audio_path(session_ref_path, paths) or session_ref_path

    # Explicit mapping for robotic/clone to session meta samples if present
    robotic_meta = _resolve_audio_path(os.path.join('meta', 'sample_robotic.wav'), paths)
    clone_meta = _resolve_audio_path(os.path.join('meta', 'sample_user.wav'), paths)

    if requested_voice.lower() == 'robotic':
        return 'robotic', (robotic_meta or default_ref)

    if requested_voice.lower() == 'clone':
        # Prioritize meta/sample_user.wav; otherwise fall back to provided ref_path; then default
        clone_ref = clone_meta or _resolve_audio_path(requested_ref, paths)
        if clone_ref:
            return 'clone', clone_ref
        log.warning(
            "Requested clone voice but no meta/sample_user.wav and ref_path='%s' not found. Falling back to %s",
            requested_ref,
            DEFAULT_VOICE_ID
        )
        return DEFAULT_VOICE_ID, default_ref

    if not requested_voice:
        return DEFAULT_VOICE_ID, default_ref

    override_ref = _resolve_audio_path(requested_ref, paths) if requested_ref else ''
    final_ref = override_ref or session_ref or default_ref
    return requested_voice, final_ref


def _append_call_log(paths: dict, record: dict):
    """Append a structured call log line to the trial directory."""
    try:
        log_path = os.path.join(paths['trial_dir'], CALL_LOG_FILENAME)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Failed to write call log: %s", e)

@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "orchestra"}), 200

@app.route('/api/v1/process', methods=['POST'])
def process_audio_pipeline():
    """
    Main endpoint for Unity to process audio through STT -> LLM -> TTS pipeline
    
    Expects multipart/form-data with:
    - audio: audio file (wav, mp3, etc.)
    - session_id: unique session identifier
    - trial_id: trial number within session
    - lang: language code (optional, default 'en')
    - voice_id: voice for TTS (optional, default 'robotic')
    - ref_path: reference audio path (required when voice_id='clone')
    - user_context: additional context for LLM (optional, inline text or path to .txt file)
    - condition: LLM response condition (optional, 1 - repeat, 2 - enhance, 3 - oppose)
    """

    start_time = time.time()
    tl = None
    timing = {
        'stt': None,
        'llm': None,
        'tts': None,
    }
    call_log_record = {
        "ts": start_time,
        "status": "unknown",
        "session_id": None,
        "trial_id": None,
        "request": {},
        "timing": {},
        "response": {}
    }

    try:
        # Extract form data
        session_id = request.form.get('session_id', f'session_{int(time.time())}')
        trial_id = int(request.form.get('trial_id', '1'))
        lang = request.form.get('lang', 'en')
        raw_voice_id = request.form.get('voice_id', '')
        raw_ref_path = request.form.get('ref_path', '')
        user_context = request.form.get('user_context', '')
        condition = request.form.get('condition', '1')  # Optional, 1 - repeat, 2 - enhance, 3 - oppose
        scene = request.form.get('scene', 'default')

        call_log_record.update({
            "session_id": session_id,
            "trial_id": trial_id,
            "request": {
                "lang": lang,
                "voice_id": raw_voice_id,
                "ref_path": raw_ref_path,
                "user_context_len": len(user_context),
                "condition": condition,
                "scene": scene
            }
        })

        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({
                "status": "error",
                "error": "No audio file provided"
            }), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                "status": "error",
                "error": "No audio file selected"
            }), 400

        log.info(f"Starting pipeline for session={session_id}, trial={trial_id}")

        # Ensure directory structure
        paths = ensure_trial_paths(session_id, trial_id)
        session_ref_candidate = os.path.join('sessions', f"{session_id}_session", 'meta', 'sample_voice.wav')
        voice_id, ref_path = _determine_voice_and_ref(raw_voice_id, raw_ref_path, session_ref_candidate, paths)

        # Initialize timeline now that we have a path to write to
        tl = Timeline(paths['timeline_path'])
        tl.add('pipeline_start', session_id=session_id, trial_id=trial_id)

        # Step 1: STT (Speech-to-Text)
        tl.add('stt_start')
        log.info("Step 1: Calling STT service...")

        stt_t0 = time.time()
        stt_files = {'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        stt_data = {
            'session_id': session_id,
            'trial_id': str(trial_id),
            'lang': lang
        }

        try:
            stt_response = requests.post(
                f"{STT_URL}/api/v1/stt",
                files=stt_files,
                data=stt_data,
                timeout=60
            )

            if stt_response.status_code != 200:
                raise Exception(f"STT service failed: {stt_response.text}")

            stt_result = stt_response.json()
            stt_text = stt_result.get('text', '')
            # get path to ASR text file if provided
            asr_text_path = stt_result.get('asr_text_path') or stt_result.get('asr_path') or ''

            log.info(f"STT completed: '{stt_text[:100]}...'")
            # record asr path and a short snippet of text in timeline
            tl.add('stt_end', asr_text_path=asr_text_path, asr_text_snippet=stt_text[:200])
            timing['stt'] = time.time() - stt_t0

        except Exception as e:
            log.error(f"STT service error: {e}")
            call_log_record["status"] = "error"
            call_log_record["timing"] = {k: v for k, v in timing.items() if v is not None}
            call_log_record["response"] = {"error": f"STT processing failed: {str(e)}"}
            _append_call_log(ensure_trial_paths(session_id, trial_id), call_log_record)
            return jsonify({
                "status": "error",
                "error": f"STT processing failed: {str(e)}",
                "timeline": tl.snapshot() if tl else []
            }), 500

        # Step 2: LLM (Language Model)
        # Construct prompt_path from STT result
        prompt_path = asr_text_path
        if prompt_path.startswith('/workspace/'):
            prompt_path = prompt_path.replace('/workspace/', '')

        llm_payload = {
            'session_id': session_id,
            'trial_id': trial_id,
            'prompt_path': prompt_path,
            'condition': str(condition),  # Optional, 1 - repeat, 2 - enhance, 3 - oppose
            'user_context': user_context,
        }
        # record the llm payload (small summary) in timeline
        tl.add('llm_start', prompt_path=prompt_path, has_user_context=bool(user_context.strip()))
        log.info("Step 2: Calling LLM service...")

        llm_t0 = time.time()
        try:
            llm_response = requests.post(
                f"{LLM_URL}/api/v1/llm",
                json=llm_payload,
                timeout=60
            )

            if llm_response.status_code != 200:
                raise Exception(f"LLM service failed: {llm_response.text}")

            llm_result = llm_response.json()
            # LLM service returns a relative path under data/ for the generated text
            llm_text_path = llm_result.get('llm_text_path') or llm_result.get('llm_text') or ''
            # try to read a small snippet from the file for the timeline if available
            llm_text = ''
            if llm_text_path:
                fp = llm_text_path
                if not fp.startswith('data' + os.sep) and not fp.startswith('data/'):
                    fp = os.path.join('data', fp)
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        llm_text = f.read().strip()
                except Exception:
                    llm_text = ''

            log.info(f"LLM completed: '{(llm_text[:100] if llm_text else llm_text_path) }...'")
            # record a small snippet of the LLM output and the path
            tl.add('llm_end', llm_text_snippet=(llm_text[:300] if llm_text else ''), llm_text_path=llm_text_path)
            timing['llm'] = time.time() - llm_t0

        except Exception as e:
            log.error(f"LLM service error: {e}")
            call_log_record["status"] = "error"
            call_log_record["timing"] = {k: v for k, v in timing.items() if v is not None}
            call_log_record["response"] = {
                "error": f"LLM processing failed: {str(e)}",
                "stt_text": stt_text
            }
            _append_call_log(ensure_trial_paths(session_id, trial_id), call_log_record)
            return jsonify({
                "status": "error",
                "error": f"LLM processing failed: {str(e)}",
                "stt_text": stt_text,
                "timeline": tl.snapshot() if tl else []
            }), 500

        # Step 3: TTS (Text-to-Speech)
        # Prepare TTS payload using the resolved voice/ref_path pair and LLM output
        tts_payload = {
            'session_id': session_id,
            'trial_id': trial_id,
            'ref_path': ref_path,  # reference audio for voice style, relative to data/, sample / robotic voice
            'text_path': llm_text_path,
        }
        tl.add('tts_start', voice_id=voice_id, ref_path=ref_path, text_path=llm_text_path)
        log.info("Step 3: Calling TTS service...")

        tts_t0 = time.time()
        try:
            tts_response = requests.post(
                f"{TTS_URL}/api/v1/tts",
                json=tts_payload,
                timeout=90
            )

            if tts_response.status_code != 200:
                raise Exception(f"TTS service failed: {tts_response.text}")

            tts_result = tts_response.json()
            tts_audio_path = tts_result.get('audio_path', tts_result.get('tts_audio_path', ''))

            log.info(f"TTS completed: {tts_audio_path}")
            tl.add('tts_end', tts_audio_path=tts_audio_path)
            timing['tts'] = time.time() - tts_t0

        except Exception as e:
            log.error(f"TTS service error: {e}")
            call_log_record["status"] = "error"
            call_log_record["timing"] = {k: v for k, v in timing.items() if v is not None}
            call_log_record["response"] = {
                "error": f"TTS processing failed: {str(e)}",
                "stt_text": stt_text,
                "llm_response": llm_text
            }
            _append_call_log(ensure_trial_paths(session_id, trial_id), call_log_record)
            return jsonify({
                "status": "error",
                "error": f"TTS processing failed: {str(e)}",
                "stt_text": stt_text,
                "llm_response": llm_text,
                "timeline": tl.snapshot() if tl else []
            }), 500

        # Calculate processing time
        total_time = time.time() - start_time
        tl.add('pipeline_end', processing_time=round(total_time, 2))

        log.info(f"Pipeline completed in {total_time:.2f}s for session={session_id}")

        call_log_record["status"] = "success"
        call_log_record["timing"] = {
            "stt": timing['stt'],
            "llm": timing['llm'],
            "tts": timing['tts'],
            "total": total_time
        }
        call_log_record["response"] = {
            "stt_text_snippet": stt_text[:200],
            "llm_response_snippet": llm_text[:200],
            "tts_audio_path": tts_audio_path,
            "voice_id": voice_id,
            "ref_path": ref_path
        }
        _append_call_log(paths, call_log_record)

        # Return comprehensive result
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "trial_id": trial_id,
            "stt_text": stt_text,
            "llm_response": llm_text,
            "tts_audio_path": tts_audio_path,
            "processing_time": round(total_time, 2),
            "lang": lang,
            "voice_id": voice_id,
            "ref_path": ref_path,
            "user_context": user_context,
            "condition": condition,
            "scene": scene,
            "timeline": tl.snapshot() if tl else []
        }), 200

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"Pipeline error: {e}\n{tb}")
        total_time = time.time() - start_time
        call_log_record["status"] = "error"
        call_log_record["timing"] = {**{k: v for k, v in timing.items()}, "total": total_time}
        call_log_record["response"] = {"error": str(e)}
        try:
            _append_call_log(paths if 'paths' in locals() else ensure_trial_paths(session_id, trial_id), call_log_record)
        except Exception:
            pass
        if tl:
            try:
                tl.add('pipeline_error', error=str(e)[:1000])
            except Exception:
                pass
            timeline_snapshot = tl.snapshot()
        else:
            timeline_snapshot = []
        # Include traceback in response for local debugging
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": tb,
            "processing_time": round(total_time, 2),
            "timeline": timeline_snapshot
        }), 500

@app.route('/api/v1/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """
    Download generated audio files for Unity
    
    Usage: GET /api/v1/download/sessions/session_id/trial_id/output.wav
    """
    try:
        file_path = os.path.join('/workspace/data', filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        if not os.path.isfile(file_path):
            return jsonify({"error": "Path is not a file"}), 400
            
        # Security check - ensure file is within data directory
        if not os.path.abspath(file_path).startswith('/workspace/data'):
            return jsonify({"error": "Access denied"}), 403
            
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        log.error(f"File download error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/upload', methods=['POST'])
def upload_file():
    """
    Lightweight upload endpoint (no pipeline execution).
    Saves an uploaded file into the requested session/trial directory.

    Form fields:
    - file: required, the file to upload
    - session_id: required, session identifier
    - trial_id: optional, defaults to 1
    - dest: optional, one of ['trial', 'meta', 'session']; defaults to 'trial'
    - filename: optional, desired filename (will be sanitized); defaults to the uploaded filename
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part provided"}), 400

        upload = request.files['file']
        if not upload.filename:
            return jsonify({"error": "Empty filename"}), 400

        session_id = request.form.get('session_id', '').strip()
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        trial_id = int(request.form.get('trial_id', '1'))

        dest_choice = request.form.get('dest', 'trial').lower()
        paths = ensure_trial_paths(session_id, trial_id)

        dest_dir = paths['trial_dir']
        if dest_choice == 'meta':
            dest_dir = paths['meta_dir']
        elif dest_choice == 'session':
            dest_dir = paths['base']

        os.makedirs(dest_dir, exist_ok=True)

        desired_name = request.form.get('filename', '').strip()
        safe_name = secure_filename(desired_name) if desired_name else secure_filename(upload.filename)
        if not safe_name:
            return jsonify({"error": "Could not derive a safe filename"}), 400

        abs_path = os.path.join(dest_dir, safe_name)
        upload.save(abs_path)

        # Build relative path from /workspace/data for convenience
        rel_path = os.path.relpath(abs_path, start='/workspace/data')

        log.info("Uploaded file saved: session=%s trial=%s dest=%s path=%s", session_id, trial_id, dest_choice, abs_path)
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "trial_id": trial_id,
            "dest": dest_choice,
            "filename": safe_name,
            "abs_path": abs_path,
            "rel_path": rel_path
        }), 200
    except Exception as e:
        log.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/status/<session_id>/<int:trial_id>', methods=['GET'])
def get_processing_status(session_id, trial_id):
    """
    Get processing status and results for a specific session/trial
    """
    try:
        paths = ensure_trial_paths(session_id, trial_id)
        
        # Check if files exist to determine processing status
        status = {
            "session_id": session_id,
            "trial_id": trial_id,
            "stt_completed": os.path.exists(os.path.join(paths['trial_dir'], 'user_1B_asr.txt')),
            "llm_completed": os.path.exists(os.path.join(paths['trial_dir'], 'user_2B_llm.txt')),
            "tts_completed": os.path.exists(os.path.join(paths['trial_dir'], 'user_2B_tts.wav')),
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        log.error(f"Status check error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    log.info("Starting Orchestra service...")
    log.info("This service orchestrates STT -> LLM -> TTS pipeline")
    
    # Run on port 7000 (different from other services)
    app.run(host='0.0.0.0', port=7000, debug=False)
