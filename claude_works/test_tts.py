"""
Quick Gemini Flash TTS test (FREE via AI Studio API key)
Model: gemini-2.5-flash-preview-tts
"""
import json
import wave
import os
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types

# ==========================================
# CONFIG
# ==========================================
CHUNK_FILE = Path("chunks_structured/ALG-AP-4/class10_math_ch05_concept_1.json")
OUTPUT_DIR = Path("claude_works/tts_test")
VOICE_NAME = "Kore"  # Options: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede
PAUSE_SECONDS = 0.4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CLIENT (free API key from aistudio.google.com)
# ==========================================
api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    print("Set GOOGLE_API_KEY first:")
    print('  $env:GOOGLE_API_KEY="your-key-from-aistudio"')
    exit(1)

client = genai.Client(api_key='AIzaSyBsEw4cqWFWMP4qdC6UYFKr_1xYrCrQxpQ')

# ==========================================
# LOAD CHUNKS
# ==========================================
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data.get("chunks", [])
print(f"Loaded {len(chunks)} chunks from {CHUNK_FILE}\n")

# ==========================================
# GENERATE TTS FOR FIRST 2 CHUNKS
# ==========================================
wav_files = []

for i, chunk in enumerate(chunks[:2]):
    chunk_id = chunk.get("chunk_id", f"chunk_{i}")
    script = chunk.get("script", "").strip()
    slide_title = chunk.get("slide_title", "")

    if not script:
        print(f"Skipping {chunk_id} - no script")
        continue

    print(f"--- Chunk {i+1}: {slide_title} ---")
    print(f"Script: {script[:120]}...")
    print(f"Voice: {VOICE_NAME}")

    out_path = OUTPUT_DIR / f"{chunk_id}.wav"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=script,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=VOICE_NAME,
                        )
                    )
                ),
            ),
        )

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
        print(f"Mime: {mime_type}, Raw size: {len(audio_data)} bytes")

        # Gemini Flash TTS returns raw PCM — wrap in WAV header
        # Default: 24000 Hz, 16-bit, mono (based on Gemini TTS docs)
        sample_rate = 24000
        sample_width = 2  # 16-bit
        channels = 1

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        # Get duration
        with wave.open(str(out_path), "rb") as wf:
            duration = wf.getnframes() / float(wf.getframerate())

        print(f"Saved: {out_path} ({out_path.stat().st_size // 1024} KB, {duration:.1f}s)")
        wav_files.append(str(out_path))

    except Exception as e:
        print(f"Error: {e}")

    print()

# ==========================================
# MERGE INTO ONE FILE
# ==========================================
if len(wav_files) >= 2:
    merged_path = OUTPUT_DIR / "preview_merged.wav"

    with wave.open(wav_files[0], "rb") as wf:
        params = wf.getparams()
        rate = wf.getframerate()

    silence = np.zeros(int(PAUSE_SECONDS * rate), dtype=np.int16).tobytes()

    with wave.open(str(merged_path), "wb") as out:
        out.setparams(params)
        for wf_path in wav_files:
            with wave.open(wf_path, "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))
            out.writeframes(silence)

    print(f"Merged preview: {merged_path}")

print("\nDone! Play with:")
print(f"  start claude_works\\tts_test\\preview_merged.wav")
