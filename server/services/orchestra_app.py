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
    - voice_id: voice for TTS (optional, default 'default')
    - user_context: additional context for LLM (optional)
    - condition: LLM response condition (optional, 1 - repeat, 2 - enhance, 3 - oppose)
    """

    start_time = time.time()
    tl = None

    try:
        # Extract form data
        session_id = request.form.get('session_id', f'session_{int(time.time())}')
        trial_id = int(request.form.get('trial_id', '1'))
        lang = request.form.get('lang', 'en')
        voice_id = request.form.get('voice_id', 'default')
        user_context = request.form.get('user_context', '')
        condition = request.form.get('condition', '1')  # Optional, 1 - repeat, 2 - enhance, 3 - oppose
        scene = request.form.get('scene', 'default')

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
        # Initialize timeline now that we have a path to write to
        tl = Timeline(paths['timeline_path'])
        tl.add('pipeline_start', session_id=session_id, trial_id=trial_id)

        # Step 1: STT (Speech-to-Text)
        tl.add('stt_start')
        log.info("Step 1: Calling STT service...")

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

        except Exception as e:
            log.error(f"STT service error: {e}")
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
        }
        # record the llm payload (small summary) in timeline
        tl.add('llm_start', prompt_path=prompt_path)
        log.info("Step 2: Calling LLM service...")

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

        except Exception as e:
            log.error(f"LLM service error: {e}")
            return jsonify({
                "status": "error",
                "error": f"LLM processing failed: {str(e)}",
                "stt_text": stt_text,
                "timeline": tl.snapshot() if tl else []
            }), 500

        # Step 3: TTS (Text-to-Speech)
        # Prepare TTS payload: ref_path derived from session_id, text_path is llm_text_path
        ref_path = os.path.join('sessions', f"{session_id}_session", 'meta', 'sample_voice.wav')
        tts_payload = {
            'session_id': session_id,
            'trial_id': trial_id,
            'ref_path': ref_path,  # reference audio for voice style, relative to data/, sample / robotic voice
            'text_path': llm_text_path,
        }
        tl.add('tts_start', voice_id=voice_id, ref_path=ref_path, text_path=llm_text_path)
        log.info("Step 3: Calling TTS service...")

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

        except Exception as e:
            log.error(f"TTS service error: {e}")
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