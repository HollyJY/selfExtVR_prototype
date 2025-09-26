"""
External test script for the three services running in Docker container.
This script sends HTTP requests to the services from outside the container.
"""

import requests
import os
import time
import json

# Service URLs (assuming Docker port mapping)
STT_URL = "http://localhost:7001"
LLM_URL = "http://localhost:7002"  
TTS_URL = "http://localhost:7003"

def test_health_checks():
    """Test health endpoints for all services"""
    print("=== Testing Health Checks ===")
    
    services = [
        ("STT", f"{STT_URL}/healthz"),
        ("LLM", f"{LLM_URL}/healthz"),
        ("TTS", f"{TTS_URL}/healthz")
    ]
    
    for name, url in services:
        try:
            resp = requests.get(url, timeout=5)
            print(f"{name} Health: {resp.status_code} - {resp.text[:50]}")
        except requests.exceptions.RequestException as e:
            print(f"{name} Health: FAILED - {e}")
    print()

def test_stt_service():
    """Test STT service with audio file"""
    print("=== Testing STT Service ===")
    
    # Use a sample audio file - adjust path as needed
    audio_file_path = "tests/test_data/sample_zjy.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        print("Please ensure you have a valid audio file for testing")
        return None
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {
                'session_id': 'test-session',
                'trial_id': '1',
                'lang': 'en'
            }
            
            resp = requests.post(f"{STT_URL}/api/v1/stt", files=files, data=data, timeout=30)
            print(f"STT Response: {resp.status_code}")
            
            if resp.status_code == 200:
                result = resp.json()
                print(f"STT Result: {json.dumps(result, indent=2)}")
                return result
            else:
                print(f"STT Error: {resp.text}")
                return None
                
    except requests.exceptions.RequestException as e:
        print(f"STT Request failed: {e}")
        return None
    except Exception as e:
        print(f"STT Test error: {e}")
        return None

def test_llm_service():
    """Test LLM service"""
    print("=== Testing LLM Service ===")
    
    payload = {
        'session_id': 'test-session',
        'trial_id': '1',
        'prompt': 'Hello, how are you?'
    }
    
    try:
        resp = requests.post(f"{LLM_URL}/api/v1/llm", json=payload, timeout=30)
        print(f"LLM Response: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"LLM Result: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"LLM Error: {resp.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"LLM Request failed: {e}")
        return None
    except Exception as e:
        print(f"LLM Test error: {e}")
        return None

def test_tts_service():
    """Test TTS service"""
    print("=== Testing TTS Service ===")
    
    payload = {
        'session_id': 'test-session',
        'trial_id': '1',
        'text': 'Hello, this is a test message for text to speech.',
        'voice_id': 'default'
    }
    
    try:
        resp = requests.post(f"{TTS_URL}/api/v1/tts", json=payload, timeout=30)
        print(f"TTS Response: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"TTS Result: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"TTS Error: {resp.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"TTS Request failed: {e}")
        return None
    except Exception as e:
        print(f"TTS Test error: {e}")
        return None

def test_full_pipeline():
    """Test the full STT -> LLM -> TTS pipeline"""
    print("=== Testing Full Pipeline ===")
    
    # Step 1: STT
    stt_result = test_stt_service()
    if not stt_result:
        print("Pipeline failed at STT step")
        return
    
    # Step 2: LLM (using STT result if available)
    llm_payload = {
        'session_id': 'test-session',
        'trial_id': '1',
        'prompt': 'Respond to this: Hello, how are you?'
    }
    
    # If STT returned text, use it in LLM prompt
    if 'text' in stt_result:
        llm_payload['prompt'] = f"User said: {stt_result['text']}. Please respond."
    
    llm_result = test_llm_service()
    if not llm_result:
        print("Pipeline failed at LLM step")
        return
    
    # Step 3: TTS (using LLM result if available)
    tts_payload = {
        'session_id': 'test-session',
        'trial_id': '1',
        'text': 'This is a test response.',
        'voice_id': 'default'
    }
    
    # If LLM returned response text, use it for TTS
    if 'response' in llm_result:
        tts_payload['text'] = llm_result['response']
    
    tts_result = test_tts_service()
    if tts_result:
        print("✅ Full pipeline test completed successfully!")
    else:
        print("Pipeline failed at TTS step")

def main():
    """Main test function"""
    print("🚀 Starting External Service Tests")
    print("Make sure Docker container is running with services started!")
    print("=" * 50)
    
    # Test health checks first
    test_health_checks()
    
    # Wait a moment
    time.sleep(1)
    
    # Test individual services
    test_stt_service()
    time.sleep(1)
    
    test_llm_service() 
    time.sleep(1)
    
    test_tts_service()
    time.sleep(1)
    
    # Test full pipeline
    test_full_pipeline()
    
    print("=" * 50)
    print("🏁 External service tests completed!")

if __name__ == "__main__":
    main()
