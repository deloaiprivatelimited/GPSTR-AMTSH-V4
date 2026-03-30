import json
import wave
import random
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech

# ==========================================
# CONFIG
# ==========================================

CHUNKS_DIR   = Path("chunks_structured")
OUTPUT_DIR   = Path("audio")
API_ENDPOINT = "texttospeech.googleapis.com"
MODEL        = "gemini-2.5-pro-tts"
MAX_WORKERS  = 3
PAUSE_SECONDS = 0.4
DEBUG        = False

# ==========================================
# VOICE OPTIONS
# ==========================================

VOICE_OPTIONS = [
    "Zephyr","Puck","Charon","Kore","Fenrir","Leda",
    "Orus","Aoede","Callirrhoe","Autonoe","Enceladus",
    "Iapetus","Umbriel","Algieba","Despina","Erinome",
    "Algenib","Rasalgethi","Laomedeia","Achernar",
    "Alnilam","Schedar","Gacrux","Pulcherrima",
    "Achird","Zubenelgenubi","Vindemiatrix",
    "Sadachbia","Sadaltager","Sulafat"
]

# ==========================================
# TTS CLIENT
# ==========================================

client = texttospeech.TextToSpeechClient(
    client_options=ClientOptions(api_endpoint=API_ENDPOINT)
)

AUDIO_CONFIG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.05,
    pitch=0
)

# ==========================================
# EXTRACT SEGMENTS FROM CHUNK
# Each chunk → one or more (text, label) pairs
# depending on tts.sync_mode
# ==========================================

def extract_segments(chunk: dict) -> list[dict]:
    """
    Returns a list of segments to synthesize for this chunk.
    Each segment: { "text": str, "label": str }

    sync_mode=chunk     → single segment from script
    sync_mode=per_step  → one segment per action_spoken
                          (solution_steps or parts[].solution_steps)
    """
    sync_mode = chunk.get("tts", {}).get("sync_mode", "chunk")
    chunk_id  = chunk.get("chunk_id", "unknown")

    if sync_mode == "chunk":
        script = chunk.get("script", "").strip()
        if not script:
            return []
        return [{"text": script, "label": f"{chunk_id}_script"}]

    # per_step — collect all action_spoken in order
    segments = []

    # single-part: solution_steps[]
    steps = chunk.get("solution_steps") or []
    for step in steps:
        text = step.get("action_spoken", "").strip()
        if text:
            segments.append({
                "text": text,
                "label": f"{chunk_id}_step_{step['step']}"
            })

    # multi-part: parts[].solution_steps[]
    parts = chunk.get("parts") or []
    for part in parts:
        for step in part.get("solution_steps", []):
            text = step.get("action_spoken", "").strip()
            if text:
                segments.append({
                    "text": text,
                    "label": f"{chunk_id}_{part['part_label']}_step_{step['step']}"
                })

    # formula derivation steps
    derivation_steps = chunk.get("derivation_steps") or []
    for step in derivation_steps:
        text = step.get("action_spoken", "").strip()
        if text:
            segments.append({
                "text": text,
                "label": f"{chunk_id}_deriv_{step['step']}"
            })

    return segments

# ==========================================
# SYNTHESIZE ONE SEGMENT
# ==========================================

def synthesize(text: str, out_path: Path, voice):
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=voice,
        audio_config=AUDIO_CONFIG
    )
    with open(out_path, "wb") as f:
        f.write(response.audio_content)


def generate_segment(seg_index: int, seg: dict, out_dir: Path, voice) -> dict:
    fname = out_dir / f"seg_{seg_index:04d}.wav"

    if not fname.exists():
        print(f"  🎙 {fname.name}")
        synthesize(seg["text"], fname, voice)

    with wave.open(str(fname), "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())

    return {
        "index":    seg_index,
        "file":     str(fname),
        "duration": duration,
        "label":    seg["label"],
        "text":     seg["text"],
    }

# ==========================================
# MERGE WAV FILES
# ==========================================

def silence_frames(duration: float, rate: int) -> bytes:
    return np.zeros(int(duration * rate), dtype=np.int16).tobytes()


def merge_wav_files(segments: list[dict], output_path: Path) -> list[dict]:
    with wave.open(segments[0]["file"], "rb") as wf:
        params = wf.getparams()
        rate   = wf.getframerate()

    timeline     = []
    current_time = 0.0

    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)

        for seg in segments:
            with wave.open(seg["file"], "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))

            timeline.append({
                "label":    seg["label"],
                "text":     seg["text"],
                "start":    round(current_time, 3),
                "end":      round(current_time + seg["duration"], 3),
                "duration": round(seg["duration"], 3),
            })

            current_time += seg["duration"] + PAUSE_SECONDS
            out.writeframes(silence_frames(PAUSE_SECONDS, rate))

    return timeline

# ==========================================
# PROCESS ONE MODULE CHUNK FILE
# ==========================================

def process_module(chunk_file: Path):
    print(f"\n📦 Processing: {chunk_file}")

    with open(chunk_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    module_id = data.get("module_id", chunk_file.stem)
    chunks    = data.get("chunks", [])

    if not chunks:
        print(f"  ⚠ No chunks found, skipping.")
        return

    # mirror chapter folder structure under audio/
    relative  = chunk_file.relative_to(CHUNKS_DIR)
    audio_dir = OUTPUT_DIR / relative.parent / module_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    final_output = audio_dir / "final_module.wav"
    timeline_out = audio_dir / "timeline.json"

    if final_output.exists():
        print(f"  ⏭ Already done, skipping.")
        return

    # pick one voice for the whole module
    voice_name = random.choice(VOICE_OPTIONS)
    print(f"  🎤 Voice: {voice_name}")

    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code="kn-IN",
        model_name=MODEL
    )

    # collect all segments across all chunks in order
    all_segments = []
    for chunk in chunks:
        all_segments.extend(extract_segments(chunk))

    if not all_segments:
        print(f"  ⚠ No speakable text found, skipping.")
        return

    print(f"  📝 {len(all_segments)} segments to synthesize")

    # synthesize in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_segment, i, seg, audio_dir, voice): i
            for i, seg in enumerate(all_segments)
        }
        results = [f.result() for f in as_completed(futures)]

    results.sort(key=lambda x: x["index"])

    # merge into one wav + write timeline
    timeline = merge_wav_files(results, final_output)

    with open(timeline_out, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Saved: {final_output}")
    print(f"  ✅ Timeline: {timeline_out}")

# ==========================================
# MAIN
# ==========================================

def main():
    chunk_files = sorted(CHUNKS_DIR.rglob("*.json"))

    if not chunk_files:
        print("No structured chunk files found.")
        return

    if DEBUG:
        chunk_files = chunk_files[:1]

    print(f"🔍 Found {len(chunk_files)} modules\n")

    for chunk_file in chunk_files:
        process_module(chunk_file)

    print("\n🎉 All modules processed.")


if __name__ == "__main__":
    main()