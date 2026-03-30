"""
Generate audio for chunks using Vertex AI Cloud TTS (paid, no preview limits)
Uses same approach as original generate_audio.py — texttospeech_v1beta1 + service account
- Only processes modules with clean chunks (zero errors) or validated ones
- Polls for new ready modules
- TTS API calls for chunks within a module run in parallel (MAX_WORKERS threads)
- Rate limiter caps requests per minute (RPM_LIMIT) to avoid 429s
"""
import json
import wave
import hashlib
import numpy as np
import time
import threading
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech

# ==========================================
# CONFIG
# ==========================================
CHUNKS_FOLDER = Path("claude_works/chunks")
VALIDATION_FOLDER = Path("claude_works/chunk_validation")
AUDIO_FOLDER = Path("claude_works/audio")

API_ENDPOINT = "texttospeech.googleapis.com"
MODEL = "gemini-2.5-pro-tts"
# MODEL = "gemini-2.5-flash-tts"

SPEAKING_RATE = 1.0
PAUSE_SECONDS = 0.6
MAX_WORKERS = 5    # max concurrent TTS API calls in flight at once
RPM_LIMIT    = 5   # max requests per minute (set below your quota, e.g. 60 → use 50)
DEBUG = False

POLL_INTERVAL = 30

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================
# TTS CLIENT (uses GOOGLE_APPLICATION_CREDENTIALS)
# ==========================================
tts_client = texttospeech.TextToSpeechClient(
    client_options=ClientOptions(api_endpoint=API_ENDPOINT)
)

AUDIO_CONFIG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=SPEAKING_RATE,
    pitch=0
)

VOICE_OPTIONS = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
    "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
    "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
    "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
    "Achird", "Zubenelgenubi", "Vindemiatrix",
    "Sadachbia", "Sadaltager", "Sulafat"
]

# ==========================================
# RATE LIMITER
# Sliding-window: tracks timestamps of the last RPM_LIMIT calls.
# Before each API call, blocks until there is room in the window.
# ==========================================
class RateLimiter:
    def __init__(self, rpm):
        self.rpm = rpm
        self.window = 60.0          # 1 minute window
        self.timestamps = deque()   # timestamps of recent calls
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                # Drop timestamps older than 1 minute
                while self.timestamps and now - self.timestamps[0] >= self.window:
                    self.timestamps.popleft()

                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return  # slot available, proceed

                # Calculate how long until the oldest slot expires
                wait = self.window - (now - self.timestamps[0])

            # Release lock while sleeping so other threads can check
            time.sleep(wait + 0.05)

rate_limiter = RateLimiter(rpm=RPM_LIMIT)

# ==========================================
# GENERATE AUDIO FOR ONE CHUNK
# ==========================================
def generate_chunk_audio(script, output_path, voice_params, retries=2):
    if output_path.exists():
        return True
    if not script or not script.strip():
        return False

    for attempt in range(retries + 1):
        try:
            rate_limiter.acquire()   # blocks if at RPM_LIMIT

            response = tts_client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=script),
                voice=voice_params,
                audio_config=AUDIO_CONFIG
            )

            with open(output_path, "wb") as f:
                f.write(response.audio_content)

            return True

        except Exception as e:
            if attempt < retries and ("429" in str(e) or "500" in str(e)):
                wait = 5 * (attempt + 1)
                print(f"    Retrying {output_path.name} in {wait}s ({e})")
                time.sleep(wait)
                continue
            print(f"    TTS error: {e}")
            return False

def get_wav_duration(wav_path):
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0

def silence_frames(duration, rate=24000):
    return np.zeros(int(duration * rate), dtype=np.int16).tobytes()

# ==========================================
# PROCESS ONE CHUNK (thread target)
# ==========================================
def process_chunk(chunk_filename, chunk_dir, audio_dir, voice_params):
    """Load chunk JSON, fire TTS API call, return metadata dict or None on failure."""
    chunk_path = chunk_dir / chunk_filename
    if not chunk_path.exists():
        return None

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    chunk_id   = chunk_data.get("chunk_id", chunk_filename.replace(".json", ""))
    script     = chunk_data.get("script", "").strip()
    slide_title = chunk_data.get("slide_title", "")
    chunk_type  = chunk_data.get("type", "")

    if not script:
        return None

    wav_path = audio_dir / f"{chunk_filename.replace('.json', '.wav')}"

    if not wav_path.exists():
        success = generate_chunk_audio(script, wav_path, voice_params)
        if not success:
            print(f"    Failed: {chunk_filename}")
            return None

    duration = get_wav_duration(wav_path)
    return {
        "chunk_id":     chunk_id,
        "chunk_file":   chunk_filename,
        "wav_file":     wav_path.name,
        "file_path":    str(wav_path),
        "duration":     round(duration, 3),
        "slide_title":  slide_title,
        "type":         chunk_type,
        "script_words": len(script.split())
    }

# ==========================================
# PROCESS ONE MODULE
# ==========================================
def process_module(merge_code, module_id):

    chunk_dir    = CHUNKS_FOLDER / merge_code / module_id
    meta_path    = chunk_dir / "_meta.json"
    audio_dir    = AUDIO_FOLDER / merge_code / module_id
    final_output = audio_dir / "final_module.wav"
    timeline_path = audio_dir / "timeline.json"

    if final_output.exists() and timeline_path.exists():
        return "skipped"

    if not meta_path.exists():
        return "no_meta"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunk_order = meta.get("chunk_order", [])
    if not chunk_order:
        return "empty"

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Pick voice deterministically
    voice_idx  = int(hashlib.md5(module_id.encode()).hexdigest(), 16) % len(VOICE_OPTIONS)
    voice_name = VOICE_OPTIONS[voice_idx]

    voice_params = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code="kn-IN",
        model_name=MODEL
    )

    print(f"  Audio: {module_id} ({len(chunk_order)} chunks, voice: {voice_name})")

    # --- Fire TTS calls in parallel (rate-limited), collect results ---
    results_by_filename = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_chunk, fn, chunk_dir, audio_dir, voice_params): fn
            for fn in chunk_order
        }
        for future in as_completed(futures):
            fn = futures[future]
            try:
                result = future.result()
                if result:
                    results_by_filename[fn] = result
            except Exception as e:
                print(f"    Error on {fn}: {e}")

    # Rebuild in original order for correct WAV merge
    chunk_wavs = [
        results_by_filename[fn]
        for fn in chunk_order
        if fn in results_by_filename
    ]

    if not chunk_wavs:
        print(f"    No audio for {module_id}")
        return "no_audio"

    # --- Merge WAVs + build timeline ---
    with wave.open(chunk_wavs[0]["file_path"], "rb") as wf:
        params = wf.getparams()
        rate   = wf.getframerate()

    timeline     = []
    current_time = 0.0

    with wave.open(str(final_output), "wb") as out:
        out.setparams(params)

        for cw in chunk_wavs:
            with wave.open(cw["file_path"], "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))

            timeline.append({
                "chunk_id":    cw["chunk_id"],
                "chunk_file":  cw["chunk_file"],
                "wav_file":    cw["wav_file"],
                "slide_title": cw["slide_title"],
                "type":        cw["type"],
                "start":       round(current_time, 3),
                "end":         round(current_time + cw["duration"], 3),
                "duration":    cw["duration"],
                "script_words": cw["script_words"]
            })

            current_time += cw["duration"]
            out.writeframes(silence_frames(PAUSE_SECONDS, rate))
            current_time += PAUSE_SECONDS

    timeline_data = {
        "module_id":               module_id,
        "voice":                   voice_name,
        "total_chunks":            len(chunk_wavs),
        "total_duration":          round(current_time, 3),
        "total_duration_formatted": f"{int(current_time//60)}:{int(current_time%60):02d}",
        "pause_between_chunks":    PAUSE_SECONDS,
        "speaking_rate":           SPEAKING_RATE,
        "segments":                timeline
    }

    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    duration_str = f"{int(current_time//60)}:{int(current_time%60):02d}"
    print(f"    OK {module_id}: {len(chunk_wavs)} chunks, {duration_str}")
    return "ok"

# ==========================================
# FIND MODULES READY FOR AUDIO
# ==========================================
def get_ready_modules():
    """Find modules ready for audio:
    - Option 1: validated (has _done.marker)
    - Option 2: chunks fully generated with ZERO errors
    """
    ready = []
    seen  = set()

    # Option 1: Validated
    if VALIDATION_FOLDER.is_dir():
        for mc in sorted(VALIDATION_FOLDER.iterdir()):
            if not mc.is_dir():
                continue
            for mod_dir in sorted(mc.iterdir()):
                if not mod_dir.is_dir():
                    continue
                if not (mod_dir / "_done.marker").exists():
                    continue

                merge_code = mc.name
                module_id  = mod_dir.name
                key        = f"{merge_code}/{module_id}"

                if (AUDIO_FOLDER / merge_code / module_id / "timeline.json").exists():
                    continue
                if not (CHUNKS_FOLDER / merge_code / module_id / "_meta.json").exists():
                    continue

                ready.append((merge_code, module_id))
                seen.add(key)

    # Option 2: Clean chunks (zero errors)
    if CHUNKS_FOLDER.is_dir():
        for mc in sorted(CHUNKS_FOLDER.iterdir()):
            if not mc.is_dir():
                continue
            for mod_dir in sorted(mc.iterdir()):
                if not mod_dir.is_dir():
                    continue

                merge_code = mc.name
                module_id  = mod_dir.name
                key        = f"{merge_code}/{module_id}"

                if key in seen:
                    continue
                if (AUDIO_FOLDER / merge_code / module_id / "timeline.json").exists():
                    continue

                meta_path = mod_dir / "_meta.json"
                if not meta_path.exists():
                    continue

                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not meta.get("errors"):
                        ready.append((merge_code, module_id))
                        seen.add(key)
                except Exception:
                    pass

    return ready

# ==========================================
# SUMMARY
# ==========================================
def generate_summary():
    print("\n" + "=" * 60)
    print("AUDIO GENERATION SUMMARY")
    print("=" * 60)

    total_modules  = 0
    total_duration = 0.0
    total_chunks   = 0
    chapter_stats  = {}

    for mc in sorted(AUDIO_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        ch_duration = 0.0
        ch_modules  = 0
        ch_chunks   = 0

        for mod_dir in sorted(mc.iterdir()):
            if not mod_dir.is_dir():
                continue
            timeline_path = mod_dir / "timeline.json"
            if timeline_path.exists():
                try:
                    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
                    ch_duration += tl.get("total_duration", 0)
                    ch_chunks   += tl.get("total_chunks", 0)
                    ch_modules  += 1
                except Exception:
                    pass

        if ch_modules:
            chapter_stats[mc.name] = {
                "modules":            ch_modules,
                "chunks":             ch_chunks,
                "duration_sec":       round(ch_duration, 1),
                "duration_formatted": f"{int(ch_duration//3600)}h {int((ch_duration%3600)//60)}m"
            }
            total_modules  += ch_modules
            total_duration += ch_duration
            total_chunks   += ch_chunks

    print(f"\nTotal modules: {total_modules}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total duration: {int(total_duration//3600)}h {int((total_duration%3600)//60)}m")

    if total_modules:
        print(f"Avg per module: {total_duration/total_modules:.0f}s")

    print(f"\nPer chapter:")
    for code, info in sorted(chapter_stats.items()):
        print(f"  {code}: {info['modules']} modules, {info['chunks']} chunks, {info['duration_formatted']}")

    summary = {
        "total_modules":            total_modules,
        "total_chunks":             total_chunks,
        "total_duration_sec":       round(total_duration, 1),
        "total_duration_formatted": f"{int(total_duration//3600)}h {int((total_duration%3600)//60)}m",
        "speaking_rate":            SPEAKING_RATE,
        "chapters":                 chapter_stats
    }

    summary_path = AUDIO_FOLDER / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")

# ==========================================
# MAIN — POLL MODE
# ==========================================
def main():
    print(f"{'='*60}")
    print(f"AUDIO GENERATION (Cloud TTS - Vertex AI)")
    print(f"{'='*60}")
    print(f"  Model:           {MODEL}")
    print(f"  Speaking rate:   {SPEAKING_RATE}")
    print(f"  Max concurrency: {MAX_WORKERS} parallel calls")
    print(f"  Rate limit:      {RPM_LIMIT} requests/min")
    print(f"  Skips validation for clean chunks (zero errors)")
    print(f"  Polls every {POLL_INTERVAL}s for new modules")
    print()

    total_processed = 0
    empty_polls     = 0

    while True:
        ready = get_ready_modules()

        if DEBUG:
            ready = ready[:1]

        if not ready:
            empty_polls += 1
            if empty_polls >= 30:
                print("No new modules for 30 polls. Finishing.")
                break
            print(f"No new modules ready. Waiting {POLL_INTERVAL}s... ({empty_polls}/30)")
            time.sleep(POLL_INTERVAL)
            continue

        empty_polls = 0
        print(f"\nFound {len(ready)} modules ready for audio")

        # Modules processed one at a time — parallelism is inside (chunk TTS calls)
        for merge_code, module_id in ready:
            process_module(merge_code, module_id)

        total_processed += len(ready)
        print(f"\nTotal processed: {total_processed} modules")

    generate_summary()
    print("\nDone")

if __name__ == "__main__":
    main()