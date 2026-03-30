"""
Multi-project parallel audio generation with schema validation.
- Validates chunk schema before TTS
- Saves _audio_meta.json per module (voice, per-chunk status)
- Failed chunks can be retried without regenerating successful ones
- Polls for new chunks from chunks_structured/
- Splits modules across N Google Cloud projects

Usage:
  python claude_works/generate_audio_multi.py
"""
import json
import wave
import hashlib
import numpy as np
import time
import threading
import os
import sys
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.oauth2 import service_account

# ==========================================
# CONFIG
# ==========================================
CHUNKS_FOLDER      = Path("claude_works/chunks_structured")
AUDIO_FOLDER       = Path("claude_works/audio_v2")
CREDENTIALS_FOLDER = Path("claude_works/credentials")

API_ENDPOINT    = "texttospeech.googleapis.com"
MODEL           = "gemini-2.5-pro-tts"

SPEAKING_RATE   = 1.0
PAUSE_SECONDS   = 0.6
RPM_PER_PROJECT = 5
POLL_INTERVAL   = 30
DEBUG           = False

# Tone prompt prepended to every script — controls voice style
TONE_PROMPT = (
    "Speak as a warm, encouraging Kannada teacher preparing students for the GPSTR exam. "
    "Use a clear, patient, and confident tone — like a favourite teacher explaining in class. "
    "Moderate pace, not too fast. Emphasize key terms slightly. "
    "Sound natural and conversational, not robotic or formal.\n\n"
)

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)

VOICE_OPTIONS = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
    "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
    "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
    "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
    "Achird", "Zubenelgenubi", "Vindemiatrix",
    "Sadachbia", "Sadaltager", "Sulafat"
]

AUDIO_CONFIG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=SPEAKING_RATE,
    pitch=0
)

# ==========================================
# REQUIRED CHUNK SCHEMA FIELDS
# ==========================================
REQUIRED_FIELDS = {"type", "slide_title", "script", "script_display", "display_bullets", "layout_config", "tts", "visual"}
REQUIRED_LAYOUT = {"layout", "template", "text_zone", "visual_zone", "transition"}
REQUIRED_TTS    = {"read_field", "language", "sync_mode"}


def validate_chunk(chunk_data, chunk_file):
    """Validate chunk matches expected schema. Returns (ok, errors)."""
    errors = []

    # Top-level required fields
    missing = REQUIRED_FIELDS - set(chunk_data.keys())
    if missing:
        errors.append(f"missing fields: {missing}")

    # layout_config
    lc = chunk_data.get("layout_config")
    if isinstance(lc, dict):
        lc_missing = REQUIRED_LAYOUT - set(lc.keys())
        if lc_missing:
            errors.append(f"layout_config missing: {lc_missing}")
    elif lc is not None:
        errors.append("layout_config is not a dict")

    # tts
    tts = chunk_data.get("tts")
    if isinstance(tts, dict):
        tts_missing = REQUIRED_TTS - set(tts.keys())
        if tts_missing:
            errors.append(f"tts missing: {tts_missing}")
    elif tts is not None:
        errors.append("tts is not a dict")

    # Script must exist and be non-empty for audio
    script = chunk_data.get("script", "")
    if not script or not script.strip():
        errors.append("script is empty")

    # Check for wrong field names (old schema variants)
    bad_fields = {"tts_config", "visual_aid", "slide_type", "script_config"}
    found_bad = bad_fields & set(chunk_data.keys())
    if found_bad:
        errors.append(f"wrong schema fields: {found_bad}")

    return (len(errors) == 0, errors)


# ==========================================
# RATE LIMITER (per-project)
# ==========================================
class RateLimiter:
    def __init__(self, rpm):
        self.rpm = rpm
        self.window = 60.0
        self.timestamps = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= self.window:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
                wait = self.window - (now - self.timestamps[0])
            time.sleep(wait + 0.05)


# ==========================================
# PROJECT WORKER
# ==========================================
class ProjectWorker:
    def __init__(self, credential_path, worker_id):
        self.worker_id = worker_id
        self.credential_path = credential_path
        self.rate_limiter = RateLimiter(rpm=RPM_PER_PROJECT)

        creds = service_account.Credentials.from_service_account_file(
            str(credential_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.tts_client = texttospeech.TextToSpeechClient(
            credentials=creds,
            client_options=ClientOptions(api_endpoint=API_ENDPOINT)
        )

        with open(credential_path, "r") as f:
            cred_data = json.load(f)
        self.project_id = cred_data.get("project_id", f"project-{worker_id}")
        self.consecutive_fails = 0
        self.dead = False
        print(f"  Worker {worker_id}: {self.project_id} ({credential_path.name})")

    def validate(self):
        try:
            self.rate_limiter.acquire()
            self.tts_client.synthesize_speech(
                input=texttospeech.SynthesisInput(text="test"),
                voice=texttospeech.VoiceSelectionParams(
                    name="Zephyr", language_code="kn-IN", model_name=MODEL
                ),
                audio_config=AUDIO_CONFIG
            )
            print(f"  Worker {self.worker_id}: OK")
            return True
        except Exception as e:
            print(f"  Worker {self.worker_id}: FAILED - {e}")
            return False

    def generate_chunk_audio(self, script, output_path, voice_params, retries=5):
        if output_path.exists():
            return True
        if not script or not script.strip():
            return False
        if self.dead:
            return False

        for attempt in range(retries + 1):
            try:
                self.rate_limiter.acquire()
                response = self.tts_client.synthesize_speech(
                    input=texttospeech.SynthesisInput(text=TONE_PROMPT + script),
                    voice=voice_params,
                    audio_config=AUDIO_CONFIG
                )
                with open(output_path, "wb") as f:
                    f.write(response.audio_content)
                self.consecutive_fails = 0
                return True
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 60 * (attempt + 1)
                    print(f"    [W{self.worker_id}] 429 on {output_path.name}, waiting {wait}s ({attempt+1}/{retries+1})")
                    time.sleep(wait)
                    continue
                elif "500" in err or "502" in err or "503" in err or "504" in err:
                    wait = 30 * (attempt + 1)
                    print(f"    [W{self.worker_id}] Server error on {output_path.name}, waiting {wait}s ({attempt+1}/{retries+1})")
                    time.sleep(wait)
                    continue
                elif "403" in err or "SERVICE_DISABLED" in err:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 3:
                        self.dead = True
                        print(f"    [W{self.worker_id}] DEAD — 3 consecutive 403s")
                    return False
                else:
                    if attempt < retries:
                        wait = 15 * (attempt + 1)
                        print(f"    [W{self.worker_id}] Error: {err[:100]}, retrying in {wait}s ({attempt+1}/{retries+1})")
                        time.sleep(wait)
                        continue
                    print(f"    [W{self.worker_id}] TTS error: {e}")
                    return False
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
# PROCESS ONE MODULE
# ==========================================
def process_module(merge_code, module_id, worker):
    chunk_dir     = CHUNKS_FOLDER / merge_code / module_id
    meta_path     = chunk_dir / "_meta.json"
    audio_dir     = AUDIO_FOLDER / merge_code / module_id
    audio_meta_path = audio_dir / "_audio_meta.json"
    final_output  = audio_dir / "final_module.wav"
    timeline_path = audio_dir / "timeline.json"

    if not meta_path.exists():
        return "no_meta"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunk_order = meta.get("chunk_order", [])
    if not chunk_order:
        return "empty"

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic voice per module
    voice_idx  = int(hashlib.md5(module_id.encode()).hexdigest(), 16) % len(VOICE_OPTIONS)
    voice_name = VOICE_OPTIONS[voice_idx]

    voice_params = texttospeech.VoiceSelectionParams(
        name=voice_name, language_code="kn-IN", model_name=MODEL
    )

    # Load existing audio meta if present (for retries)
    existing_audio_meta = {}
    if audio_meta_path.exists():
        try:
            existing_audio_meta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
            # If voice changed (shouldn't happen but safety), wipe everything
            if existing_audio_meta.get("voice") != voice_name:
                existing_audio_meta = {}
        except Exception:
            existing_audio_meta = {}

    existing_chunks = existing_audio_meta.get("chunks", {})

    # ---- Step 1: Validate all chunks and collect work ----
    valid_chunks = []       # (filename, chunk_data) for chunks with valid schema + script
    schema_errors = []      # chunks that failed validation

    for fn in chunk_order:
        chunk_path = chunk_dir / fn
        if not chunk_path.exists():
            continue

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        ok, errs = validate_chunk(chunk_data, fn)
        script = chunk_data.get("script", "").strip()

        if not ok:
            schema_errors.append({"file": fn, "errors": errs})
            continue

        if not script:
            continue

        valid_chunks.append((fn, chunk_data))

    if schema_errors:
        print(f"  [W{worker.worker_id}] {module_id}: {len(schema_errors)} chunks failed schema validation:")
        for se in schema_errors[:3]:
            print(f"    {se['file']}: {se['errors']}")
        if len(schema_errors) > 3:
            print(f"    ... and {len(schema_errors)-3} more")

    if not valid_chunks:
        print(f"  [W{worker.worker_id}] {module_id}: no valid chunks, skipping")
        return "no_valid_chunks"

    # ---- Step 2: Figure out which chunks need TTS ----
    chunks_needing_tts = []
    chunks_already_done = []

    for fn, chunk_data in valid_chunks:
        wav_path = audio_dir / fn.replace(".json", ".wav")
        script = chunk_data.get("script", "").strip()
        script_hash = hashlib.md5(script.encode()).hexdigest()

        prev = existing_chunks.get(fn, {})

        # WAV exists AND script hasn't changed AND same voice → skip
        if (wav_path.exists()
            and prev.get("status") == "ok"
            and prev.get("script_hash") == script_hash
            and prev.get("voice") == voice_name):
            chunks_already_done.append(fn)
        else:
            # Delete stale WAV if script changed
            if wav_path.exists() and prev.get("script_hash") != script_hash:
                wav_path.unlink()
            chunks_needing_tts.append((fn, chunk_data, script_hash))

    # If everything done and final exists, skip
    if not chunks_needing_tts and final_output.exists() and timeline_path.exists():
        return "skipped"

    if chunks_needing_tts:
        print(f"  [W{worker.worker_id}] {module_id}: {len(chunks_needing_tts)} to generate, {len(chunks_already_done)} cached (voice: {voice_name})")
    else:
        print(f"  [W{worker.worker_id}] {module_id}: all WAVs cached, rebuilding final (voice: {voice_name})")

    # ---- Step 3: Generate TTS for pending chunks ----
    chunk_results = {}  # fn -> status dict

    # Copy over existing successful results
    for fn in chunks_already_done:
        chunk_results[fn] = existing_chunks[fn]

    def tts_one_chunk(fn, chunk_data, script_hash):
        script = chunk_data.get("script", "").strip()
        wav_path = audio_dir / fn.replace(".json", ".wav")

        success = worker.generate_chunk_audio(script, wav_path, voice_params)

        if success and wav_path.exists():
            duration = get_wav_duration(wav_path)
            return fn, {
                "status": "ok",
                "voice": voice_name,
                "script_hash": script_hash,
                "wav_file": wav_path.name,
                "duration": round(duration, 3),
                "slide_title": chunk_data.get("slide_title", ""),
                "type": chunk_data.get("type", ""),
                "script_words": len(script.split()),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            return fn, {
                "status": "failed",
                "voice": voice_name,
                "script_hash": script_hash,
                "error": "TTS call failed after retries",
                "slide_title": chunk_data.get("slide_title", ""),
                "type": chunk_data.get("type", ""),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    # Run TTS calls in parallel within this worker's rate limiter
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(tts_one_chunk, fn, cd, sh): fn
            for fn, cd, sh in chunks_needing_tts
        }
        for future in as_completed(futures):
            try:
                fn, result = future.result()
                chunk_results[fn] = result
            except Exception as e:
                fn = futures[future]
                chunk_results[fn] = {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

    # ---- Step 4: Save _audio_meta.json (always, even if some failed) ----
    ok_count = sum(1 for v in chunk_results.values() if v.get("status") == "ok")
    fail_count = sum(1 for v in chunk_results.values() if v.get("status") == "failed")

    audio_meta = {
        "module_id": module_id,
        "merge_code": merge_code,
        "voice": voice_name,
        "model": MODEL,
        "speaking_rate": SPEAKING_RATE,
        "total_chunks": len(valid_chunks),
        "ok_count": ok_count,
        "failed_count": fail_count,
        "schema_errors": len(schema_errors),
        "schema_error_files": [se["file"] for se in schema_errors],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": chunk_results,
    }

    with open(audio_meta_path, "w", encoding="utf-8") as f:
        json.dump(audio_meta, f, indent=2, ensure_ascii=False)

    # ---- Step 5: If any failed, don't build final ----
    if fail_count > 0:
        print(f"    [W{worker.worker_id}] {module_id}: {fail_count} chunks FAILED, skipping final merge. Re-run to retry.")
        return "partial"

    # ---- Step 6: Build final WAV + timeline ----
    # Collect WAVs in order
    ordered_wavs = []
    for fn, chunk_data in valid_chunks:
        info = chunk_results.get(fn)
        if not info or info.get("status") != "ok":
            continue
        wav_path = audio_dir / info["wav_file"]
        if not wav_path.exists():
            continue
        ordered_wavs.append({
            "chunk_id": chunk_data.get("chunk_id", fn.replace(".json", "")),
            "chunk_file": fn,
            "wav_file": info["wav_file"],
            "file_path": str(wav_path),
            "duration": info["duration"],
            "slide_title": info.get("slide_title", ""),
            "type": info.get("type", ""),
            "script_words": info.get("script_words", 0),
        })

    if not ordered_wavs:
        return "no_audio"

    # Merge
    with wave.open(ordered_wavs[0]["file_path"], "rb") as wf:
        params = wf.getparams()
        rate = wf.getframerate()

    timeline = []
    current_time = 0.0

    with wave.open(str(final_output), "wb") as out:
        out.setparams(params)
        for cw in ordered_wavs:
            with wave.open(cw["file_path"], "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))
            timeline.append({
                "chunk_id":     cw["chunk_id"],
                "chunk_file":   cw["chunk_file"],
                "wav_file":     cw["wav_file"],
                "slide_title":  cw["slide_title"],
                "type":         cw["type"],
                "start":        round(current_time, 3),
                "end":          round(current_time + cw["duration"], 3),
                "duration":     cw["duration"],
                "script_words": cw["script_words"],
            })
            current_time += cw["duration"]
            out.writeframes(silence_frames(PAUSE_SECONDS, rate))
            current_time += PAUSE_SECONDS

    timeline_data = {
        "module_id":                module_id,
        "voice":                    voice_name,
        "model":                    MODEL,
        "total_chunks":             len(ordered_wavs),
        "total_duration":           round(current_time, 3),
        "total_duration_formatted": f"{int(current_time//60)}:{int(current_time%60):02d}",
        "pause_between_chunks":     PAUSE_SECONDS,
        "speaking_rate":            SPEAKING_RATE,
        "segments":                 timeline,
    }

    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    duration_str = f"{int(current_time//60)}:{int(current_time%60):02d}"
    print(f"    [W{worker.worker_id}] OK {module_id}: {len(ordered_wavs)} chunks, {duration_str}")
    return "ok"


# ==========================================
# FIND MODULES READY FOR AUDIO
# ==========================================
def get_ready_modules():
    """Find modules that have chunks and need audio (or have failed chunks to retry)."""
    ready = []

    if not CHUNKS_FOLDER.is_dir():
        return ready

    for mc in sorted(CHUNKS_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        for mod_dir in sorted(mc.iterdir()):
            if not mod_dir.is_dir():
                continue

            merge_code = mc.name
            module_id  = mod_dir.name

            meta_path = mod_dir / "_meta.json"
            if not meta_path.exists():
                continue

            # Check if chunks generated without errors
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("errors"):
                    continue  # chunks not fully generated
            except Exception:
                continue

            audio_dir = AUDIO_FOLDER / merge_code / module_id
            audio_meta_path = audio_dir / "_audio_meta.json"
            timeline_path = audio_dir / "timeline.json"

            # Case 1: No audio at all yet → needs processing
            if not audio_meta_path.exists():
                ready.append((merge_code, module_id))
                continue

            # Case 2: Has audio meta — check for failures to retry
            try:
                ameta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
                if ameta.get("failed_count", 0) > 0:
                    ready.append((merge_code, module_id))
                    continue
                # Case 3: All chunks OK but no final WAV
                if not timeline_path.exists():
                    ready.append((merge_code, module_id))
                    continue
            except Exception:
                ready.append((merge_code, module_id))

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
    total_failed   = 0
    chapter_stats  = {}

    for mc in sorted(AUDIO_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        ch_duration = 0.0
        ch_modules  = 0
        ch_chunks   = 0
        ch_failed   = 0

        for mod_dir in sorted(mc.iterdir()):
            if not mod_dir.is_dir():
                continue

            audio_meta_path = mod_dir / "_audio_meta.json"
            timeline_path = mod_dir / "timeline.json"

            if audio_meta_path.exists():
                try:
                    ameta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
                    ch_modules += 1
                    ch_chunks += ameta.get("ok_count", 0)
                    ch_failed += ameta.get("failed_count", 0)
                except Exception:
                    pass

            if timeline_path.exists():
                try:
                    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
                    ch_duration += tl.get("total_duration", 0)
                except Exception:
                    pass

        if ch_modules:
            chapter_stats[mc.name] = {
                "modules":            ch_modules,
                "chunks_ok":          ch_chunks,
                "chunks_failed":      ch_failed,
                "duration_sec":       round(ch_duration, 1),
                "duration_formatted": f"{int(ch_duration//3600)}h {int((ch_duration%3600)//60)}m",
            }
            total_modules  += ch_modules
            total_duration += ch_duration
            total_chunks   += ch_chunks
            total_failed   += ch_failed

    print(f"\nTotal modules: {total_modules}")
    print(f"Total chunks OK: {total_chunks}")
    print(f"Total chunks FAILED: {total_failed}")
    print(f"Total duration: {int(total_duration//3600)}h {int((total_duration%3600)//60)}m")

    if total_modules:
        print(f"Avg per module: {total_duration/total_modules:.0f}s")

    print(f"\nPer chapter:")
    for code, info in sorted(chapter_stats.items()):
        fail_str = f", {info['chunks_failed']} failed" if info['chunks_failed'] else ""
        print(f"  {code}: {info['modules']} modules, {info['chunks_ok']} chunks{fail_str}, {info['duration_formatted']}")

    summary = {
        "total_modules":            total_modules,
        "total_chunks_ok":          total_chunks,
        "total_chunks_failed":      total_failed,
        "total_duration_sec":       round(total_duration, 1),
        "total_duration_formatted": f"{int(total_duration//3600)}h {int((total_duration%3600)//60)}m",
        "speaking_rate":            SPEAKING_RATE,
        "model":                    MODEL,
        "chapters":                 chapter_stats,
    }

    summary_path = AUDIO_FOLDER / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


# ==========================================
# MAIN
# ==========================================
def main():
    cred_files = sorted(CREDENTIALS_FOLDER.glob("*.json"))

    env_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_cred and Path(env_cred).exists():
        env_path = Path(env_cred)
        if env_path not in cred_files:
            cred_files.insert(0, env_path)

    if not cred_files:
        print("ERROR: No credential files found!")
        print(f"  Put service account JSON files in: {CREDENTIALS_FOLDER}/")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"MULTI-PROJECT AUDIO GENERATION")
    print(f"{'='*60}")
    print(f"  Model:           {MODEL}")
    print(f"  Speaking rate:   {SPEAKING_RATE}")
    print(f"  Chunks from:     {CHUNKS_FOLDER}")
    print(f"  Projects:        {len(cred_files)}")
    print(f"  RPM per project: {RPM_PER_PROJECT}")
    print(f"  Effective RPM:   {RPM_PER_PROJECT * len(cred_files)}")
    print()

    # Create workers
    workers = []
    for i, cred_path in enumerate(cred_files):
        try:
            w = ProjectWorker(cred_path, i)
            workers.append(w)
        except Exception as e:
            print(f"  SKIP {cred_path.name}: {e}")

    if not workers:
        print("ERROR: No valid workers!")
        sys.exit(1)

    # Validate
    print(f"\n  Validating {len(workers)} workers...")
    valid_workers = [w for w in workers if w.validate()]

    if not valid_workers:
        print("ERROR: No workers passed validation!")
        sys.exit(1)

    workers = valid_workers
    print(f"\n  Active workers: {len(workers)}")
    print(f"  Effective RPM:  {RPM_PER_PROJECT * len(workers)}")
    print()

    total_processed = 0
    empty_polls     = 0

    while True:
        workers = [w for w in workers if not w.dead]
        if not workers:
            print("ERROR: All workers are dead!")
            break

        ready = get_ready_modules()

        if DEBUG:
            ready = ready[:len(workers)]

        if not ready:
            empty_polls += 1
            if empty_polls >= 60:
                print("No new modules for 60 polls. Finishing.")
                break
            print(f"No new modules ready. Waiting {POLL_INTERVAL}s... ({empty_polls}/60)")
            time.sleep(POLL_INTERVAL)
            continue

        empty_polls = 0
        print(f"\nFound {len(ready)} modules ready for audio (workers: {len(workers)})")

        # Split round-robin across workers
        worker_assignments = [[] for _ in workers]
        for i, (mc, mid) in enumerate(ready):
            worker_assignments[i % len(workers)].append((mc, mid))

        def worker_run(w, modules):
            for mc, mid in modules:
                if w.dead:
                    break
                process_module(mc, mid, w)

        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = []
            for w, modules in zip(workers, worker_assignments):
                if modules:
                    futures.append(executor.submit(worker_run, w, modules))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Worker error: {e}")

        total_processed += len(ready)
        print(f"\nTotal processed: {total_processed} modules")

    generate_summary()
    print("\nDone")


if __name__ == "__main__":
    main()
