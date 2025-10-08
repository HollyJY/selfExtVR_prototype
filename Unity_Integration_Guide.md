# Unity Integration Guide for Orchestra Service

## Overview
The Orchestra service provides a single endpoint for Unity to process audio through the complete STT → LLM → TTS pipeline.

## Service Endpoint
**Base URL**: `http://your-server-ip:7000`

## Main Processing Endpoint

### POST `/api/v1/process`
Process audio through the full pipeline.

**Content-Type**: `multipart/form-data`

**Parameters**:
- `audio` (file): Audio file (WAV, MP3, etc.)
- `session_id` (string): Unique session identifier
- `trial_id` (int): Trial number within session
- `lang` (string, optional): Language code (default: 'en')
- `voice_id` (string, optional): Voice ID for TTS (default: 'default')
- `user_context` (string, optional): Additional context for LLM

**Response** (JSON):
```json
{
  "status": "success",
  "session_id": "your_session_id",
  "trial_id": 1,
  "stt_text": "transcribed text from audio",
  "llm_response": "AI generated response",
  "tts_audio_path": "sessions/your_session/trial_001/user_2B_tts.wav",
  "processing_time": 5.23,
  "lang": "en",
  "voice_id": "default",
  "timeline": {
    "pipeline_start": 1234567890.123,
    "stt_start": 1234567890.200,
    "stt_end": 1234567892.100,
    "llm_start": 1234567892.150,
    "llm_end": 1234567894.300,
    "tts_start": 1234567894.350,
    "tts_end": 1234567896.800,
    "pipeline_end": 1234567896.850
  }
}
```

## Download Audio File

### GET `/api/v1/download/<path>`
Download the generated TTS audio file.

**Example**: `GET /api/v1/download/sessions/your_session/trial_001/user_2B_tts.wav`

## Unity C# Example

```csharp
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class AudioProcessor : MonoBehaviour
{
    private string serverUrl = "http://your-server-ip:7000";
    
    public IEnumerator ProcessAudio(byte[] audioData, string sessionId, int trialId)
    {
        // Create form data
        WWWForm form = new WWWForm();
        form.AddBinaryData("audio", audioData, "recording.wav", "audio/wav");
        form.AddField("session_id", sessionId);
        form.AddField("trial_id", trialId.ToString());
        form.AddField("lang", "en");
        form.AddField("voice_id", "default");
        form.AddField("user_context", "This is a VR conversation");
        
        // Send request
        using (UnityWebRequest request = UnityWebRequest.Post($"{serverUrl}/api/v1/process", form))
        {
            request.timeout = 120; // 2 minutes timeout
            
            Debug.Log("Sending audio to orchestra service...");
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                string jsonResponse = request.downloadHandler.text;
                ProcessingResult result = JsonUtility.FromJson<ProcessingResult>(jsonResponse);
                
                if (result.status == "success")
                {
                    Debug.Log($"STT: {result.stt_text}");
                    Debug.Log($"LLM: {result.llm_response}");
                    Debug.Log($"Processing time: {result.processing_time}s");
                    
                    // Download the generated audio
                    yield return StartCoroutine(DownloadAudio(result.tts_audio_path));
                }
                else
                {
                    Debug.LogError($"Processing failed: {result.error}");
                }
            }
            else
            {
                Debug.LogError($"Request failed: {request.error}");
            }
        }
    }
    
    public IEnumerator DownloadAudio(string audioPath)
    {
        string downloadUrl = $"{serverUrl}/api/v1/download/{audioPath}";
        
        using (UnityWebRequest request = UnityWebRequest.Get(downloadUrl))
        {
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                byte[] audioData = request.downloadHandler.data;
                
                // Save to Unity's persistent data path
                string filePath = System.IO.Path.Combine(
                    Application.persistentDataPath, 
                    "response_audio.wav"
                );
                System.IO.File.WriteAllBytes(filePath, audioData);
                
                Debug.Log($"Audio saved to: {filePath}");
                
                // Play the audio in Unity
                yield return StartCoroutine(PlayAudioFile(filePath));
            }
            else
            {
                Debug.LogError($"Audio download failed: {request.error}");
            }
        }
    }
    
    private IEnumerator PlayAudioFile(string filePath)
    {
        using (UnityWebRequest request = UnityWebRequestMultimedia.GetAudioClip($"file://{filePath}", AudioType.WAV))
        {
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(request);
                AudioSource audioSource = GetComponent<AudioSource>();
                audioSource.clip = clip;
                audioSource.Play();
                
                Debug.Log("Playing AI response audio");
            }
        }
    }
}

[System.Serializable]
public class ProcessingResult
{
    public string status;
    public string session_id;
    public int trial_id;
    public string stt_text;
    public string llm_response;
    public string tts_audio_path;
    public float processing_time;
    public string lang;
    public string voice_id;
    public string error;
    // timeline object can be added if needed
}
```

## Health Check
### GET `/healthz`
Check if the orchestra service is running.

## Status Check
### GET `/api/v1/status/<session_id>/<trial_id>`
Check processing status for a specific session/trial.

## Setup Instructions

1. **Build Docker Container** (if not already built):
   ```bash
   docker build -t extselfvr:ollama server
   ```

2. **Run Container with Orchestra Service**:
   ```bash
   chmod +x run_orchestra.sh
   ./run_orchestra.sh
   ```

3. **Wait for Services to Start** (check logs):
   ```bash
   docker exec -it <container_id> tail -f /workspace/logs/orchestra.log
   ```

4. **Test from Unity**: Use the C# code above or test with curl:
   ```bash
   curl -X POST -F "audio=@test.wav" -F "session_id=test" -F "trial_id=1" \
        http://your-server-ip:7000/api/v1/process
   ```

## Notes
- The orchestra service runs on port 7000
- Processing typically takes 5-15 seconds depending on audio length
- All files are organized by session_id and trial_id
- The service provides detailed timing information for performance monitoring
- Audio files are automatically cleaned up after processing (configurable)