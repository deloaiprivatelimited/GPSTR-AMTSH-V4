"""
Generate podcast audio from two-person dialogue scripts.
- Reads podcast JSONs from claude_works/podcasts/{MERGE_CODE}/{MODULE_ID}_podcast.json
- Teacher = fixed MALE voice, Student = fixed FEMALE voice (consistent across all modules)
- Each dialogue turn → one TTS call → one WAV
- All turns merged into one final podcast WAV per module
- Multi-project parallel with rate limiting (same pattern as generate_audio_multi.py)
- Skips already-done modules, retries failed turns on re-run

Usage:
  python claude_works/generate_podcast_audio.py
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
PODCASTS_FOLDER    = Path("claude_works/podcasts")
AUDIO_FOLDER       = Path("claude_works/podcast_audio1")
CREDENTIALS_FOLDER = Path("claude_works/credentials")

API_ENDPOINT    = "texttospeech.googleapis.com"
MODEL           = "gemini-2.5-pro-tts"

SPEAKING_RATE   = 1.1
PAUSE_SAME_SPEAKER   = 0.3   # short pause when same speaker continues
PAUSE_SPEAKER_SWITCH = 0.6   # slightly longer on speaker change
RPM_PER_PROJECT = 5
POLL_INTERVAL   = 30
DEBUG           = True

# ==========================================
# VOICE ASSIGNMENT — consistent across ALL modules
# ==========================================
# Male voice for teacher, female voice for student
TEACHER_VOICE = "Puck"       # male, warm, clear
STUDENT_VOICE = "Kore"       # female, young, curious

TEACHER_TONE = (
    "Speak as a warm, experienced Kannada maths teacher. "
    "Clear, patient, confident tone. Moderate pace, slightly slower on key terms. "
    "Sound like a favourite teacher explaining during a free period. "
    "Natural and conversational, not robotic.\n\n"
)

STUDENT_TONE = (
    "Speak as a curious, engaged young Kannada student preparing for an exam. "
    "Enthusiastic but natural. Sometimes surprised, sometimes confirming understanding. "
    "Sound like a smart student in a one-on-one revision session.\n\n"
)

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)

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
                    name=TEACHER_VOICE, language_code="kn-IN", model_name=MODEL
                ),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    speaking_rate=SPEAKING_RATE, pitch=0
                )
            )
            print(f"  Worker {self.worker_id}: OK")
            return True
        except Exception as e:
            print(f"  Worker {self.worker_id}: FAILED - {e}")
            return False

    def generate_audio(self, script, output_path, voice_name, tone_prompt, retries=5):
        """Generate TTS for one dialogue turn."""
        if output_path.exists():
            return True
        if not script or not script.strip():
            return False
        if self.dead:
            return False

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=SPEAKING_RATE,
            pitch=0
        )
        voice_params = texttospeech.VoiceSelectionParams(
            name=voice_name, language_code="kn-IN", model_name=MODEL
        )

        for attempt in range(retries + 1):
            try:
                self.rate_limiter.acquire()
                response = self.tts_client.synthesize_speech(
                    input=texttospeech.SynthesisInput(text=tone_prompt + script),
                    voice=voice_params,
                    audio_config=audio_config
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
                elif "500" in err or "502" in err or "503" in err or "504" in err:
                    wait = 30 * (attempt + 1)
                    print(f"    [W{self.worker_id}] Server error, waiting {wait}s ({attempt+1}/{retries+1})")
                    time.sleep(wait)
                elif "403" in err or "SERVICE_DISABLED" in err:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 3:
                        self.dead = True
                        print(f"    [W{self.worker_id}] DEAD — 3 consecutive 403s")
                    return False
                else:
                    if attempt < retries:
                        wait = 15 * (attempt + 1)
                        print(f"    [W{self.worker_id}] Error: {err[:100]}, retrying in {wait}s")
                        time.sleep(wait)
                    else:
                        print(f"    [W{self.worker_id}] TTS error: {e}")
                        return False
        return False

# ==========================================
# HELPERS
# ==========================================
def get_wav_duration(wav_path):
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0

def silence_frames(duration, rate=24000):
    return np.zeros(int(duration * rate), dtype=np.int16).tobytes()

# ==========================================
# PROCESS ONE PODCAST MODULE
# ==========================================
def process_module(merge_code, podcast_file, worker):
    podcast_path = PODCASTS_FOLDER / merge_code / podcast_file
    module_id = podcast_file.replace("_podcast.json", "")

    audio_dir       = AUDIO_FOLDER / merge_code / module_id
    audio_meta_path = audio_dir / "_audio_meta.json"
    final_output    = audio_dir / "podcast_final.wav"
    timeline_path   = audio_dir / "timeline.json"

    # Load podcast JSON
    try:
        podcast_data = json.loads(podcast_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [W{worker.worker_id}] Cannot read {podcast_file}: {e}")
        return "error"

    dialogues = podcast_data.get("dialogues", [])
    if not dialogues:
        print(f"  [W{worker.worker_id}] {module_id}: no dialogues, skipping")
        return "empty"

    # Check for generation errors from podcast script gen
    if podcast_data.get("errors"):
        print(f"  [W{worker.worker_id}] {module_id}: podcast has errors, skipping")
        return "error"

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load existing audio meta for retry logic
    existing_meta = {}
    if audio_meta_path.exists():
        try:
            existing_meta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing_turns = existing_meta.get("turns", {})

    # Figure out which turns need TTS
    turns_needing_tts = []
    turns_already_done = []

    for d in dialogues:
        idx = d["index"]
        speaker = d["speaker"]
        script = d.get("script", "").strip()
        if not script:
            continue

        wav_name = f"{idx:03d}_{speaker}.wav"
        wav_path = audio_dir / wav_name
        script_hash = hashlib.md5(script.encode()).hexdigest()

        prev = existing_turns.get(str(idx), {})

        # Already done if WAV exists, script unchanged, same voices
        if (wav_path.exists()
            and prev.get("status") == "ok"
            and prev.get("script_hash") == script_hash):
            turns_already_done.append(str(idx))
        else:
            # Delete stale WAV
            if wav_path.exists() and prev.get("script_hash") != script_hash:
                wav_path.unlink()
            turns_needing_tts.append((d, wav_name, script_hash))

    # Skip if everything done and final exists
    if not turns_needing_tts and final_output.exists() and timeline_path.exists():
        return "skipped"

    if turns_needing_tts:
        print(f"  [W{worker.worker_id}] {module_id}: {len(turns_needing_tts)} to generate, {len(turns_already_done)} cached")
    else:
        print(f"  [W{worker.worker_id}] {module_id}: all cached, rebuilding final")

    # Copy over existing results
    turn_results = {}
    for idx_str in turns_already_done:
        turn_results[idx_str] = existing_turns[idx_str]

    # Generate TTS for pending turns
    def tts_one_turn(dialogue, wav_name, script_hash):
        idx = dialogue["index"]
        speaker = dialogue["speaker"]
        script = dialogue["script"].strip()
        section = dialogue.get("section", "")

        voice_name = TEACHER_VOICE if speaker == "teacher" else STUDENT_VOICE
        tone = TEACHER_TONE if speaker == "teacher" else STUDENT_TONE

        wav_path = audio_dir / wav_name

        success = worker.generate_audio(script, wav_path, voice_name, tone)

        if success and wav_path.exists():
            duration = get_wav_duration(wav_path)
            return str(idx), {
                "status": "ok",
                "index": idx,
                "speaker": speaker,
                "voice": voice_name,
                "section": section,
                "script_hash": script_hash,
                "wav_file": wav_name,
                "duration": round(duration, 3),
                "script_words": len(script.split()),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            return str(idx), {
                "status": "failed",
                "index": idx,
                "speaker": speaker,
                "voice": voice_name,
                "section": section,
                "script_hash": script_hash,
                "error": "TTS failed after retries",
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    # Parallel TTS within worker's rate limiter
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(tts_one_turn, d, wn, sh): d["index"]
            for d, wn, sh in turns_needing_tts
        }
        for future in as_completed(futures):
            try:
                idx_str, result = future.result()
                turn_results[idx_str] = result
            except Exception as e:
                idx = futures[future]
                turn_results[str(idx)] = {
                    "status": "failed",
                    "index": idx,
                    "error": str(e),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

    # Save audio meta
    ok_count = sum(1 for v in turn_results.values() if v.get("status") == "ok")
    fail_count = sum(1 for v in turn_results.values() if v.get("status") == "failed")

    audio_meta = {
        "module_id": module_id,
        "merge_code": merge_code,
        "teacher_voice": TEACHER_VOICE,
        "student_voice": STUDENT_VOICE,
        "model": MODEL,
        "speaking_rate": SPEAKING_RATE,
        "total_turns": len(dialogues),
        "ok_count": ok_count,
        "failed_count": fail_count,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "turns": turn_results,
    }

    audio_meta_path.write_text(json.dumps(audio_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # If any failed, don't build final
    if fail_count > 0:
        print(f"    [W{worker.worker_id}] {module_id}: {fail_count} turns FAILED, re-run to retry")
        return "partial"

    # Build final WAV + timeline — in dialogue order
    ordered_wavs = []
    for d in dialogues:
        idx_str = str(d["index"])
        info = turn_results.get(idx_str)
        if not info or info.get("status") != "ok":
            continue
        wav_path = audio_dir / info["wav_file"]
        if not wav_path.exists():
            continue
        ordered_wavs.append({
            "index": d["index"],
            "speaker": d["speaker"],
            "section": d.get("section", ""),
            "wav_file": info["wav_file"],
            "file_path": str(wav_path),
            "duration": info["duration"],
            "script_words": info.get("script_words", 0),
        })

    if not ordered_wavs:
        return "no_audio"

    # Get sample params from first WAV
    with wave.open(ordered_wavs[0]["file_path"], "rb") as wf:
        params = wf.getparams()
        rate = wf.getframerate()

    timeline = []
    current_time = 0.0

    with wave.open(str(final_output), "wb") as out:
        out.setparams(params)

        prev_speaker = None
        for cw in ordered_wavs:
            # Write audio frames
            with wave.open(cw["file_path"], "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))

            timeline.append({
                "index":        cw["index"],
                "speaker":      cw["speaker"],
                "section":      cw["section"],
                "wav_file":     cw["wav_file"],
                "start":        round(current_time, 3),
                "end":          round(current_time + cw["duration"], 3),
                "duration":     cw["duration"],
                "script_words": cw["script_words"],
            })

            current_time += cw["duration"]

            # Smart pause: shorter for same speaker, longer on switch
            if cw["speaker"] == prev_speaker:
                pause = PAUSE_SAME_SPEAKER
            else:
                pause = PAUSE_SPEAKER_SWITCH

            out.writeframes(silence_frames(pause, rate))
            current_time += pause
            prev_speaker = cw["speaker"]

    # Count per-speaker stats
    teacher_time = sum(s["duration"] for s in timeline if s["speaker"] == "teacher")
    student_time = sum(s["duration"] for s in timeline if s["speaker"] == "student")

    timeline_data = {
        "module_id":                module_id,
        "module_title":             podcast_data.get("module_title", ""),
        "teacher_voice":            TEACHER_VOICE,
        "student_voice":            STUDENT_VOICE,
        "model":                    MODEL,
        "speaking_rate":            SPEAKING_RATE,
        "total_turns":              len(ordered_wavs),
        "total_duration":           round(current_time, 3),
        "total_duration_formatted": f"{int(current_time//60)}:{int(current_time%60):02d}",
        "teacher_duration":         round(teacher_time, 3),
        "student_duration":         round(student_time, 3),
        "pause_same_speaker":       PAUSE_SAME_SPEAKER,
        "pause_speaker_switch":     PAUSE_SPEAKER_SWITCH,
        "estimated_duration_from_script": podcast_data.get("podcast_meta", {}).get("estimated_duration_minutes", 0),
        "segments":                 timeline,
    }

    timeline_path.write_text(json.dumps(timeline_data, indent=2, ensure_ascii=False), encoding="utf-8")

    duration_str = f"{int(current_time//60)}:{int(current_time%60):02d}"
    print(f"    [W{worker.worker_id}] OK {module_id}: {len(ordered_wavs)} turns, {duration_str} "
          f"(teacher:{int(teacher_time)}s student:{int(student_time)}s)")
    return "ok"

# ==========================================
# FIND MODULES READY FOR AUDIO
# ==========================================
def get_ready_modules():
    ready = []
    if not PODCASTS_FOLDER.is_dir():
        return ready

    for mc in sorted(PODCASTS_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        for f in sorted(mc.iterdir()):
            if not f.name.endswith("_podcast.json"):
                continue

            merge_code = mc.name
            module_id = f.name.replace("_podcast.json", "")

            # Check podcast has dialogues and no errors
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not data.get("dialogues") or data.get("errors"):
                    continue
            except Exception:
                continue

            audio_dir = AUDIO_FOLDER / merge_code / module_id
            audio_meta_path = audio_dir / "_audio_meta.json"
            timeline_path = audio_dir / "timeline.json"

            # Case 1: No audio yet
            if not audio_meta_path.exists():
                ready.append((merge_code, f.name))
                continue

            # Case 2: Has failures to retry
            try:
                ameta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
                if ameta.get("failed_count", 0) > 0:
                    ready.append((merge_code, f.name))
                    continue
                # Case 3: All OK but no final
                if not timeline_path.exists():
                    ready.append((merge_code, f.name))
                    continue
            except Exception:
                ready.append((merge_code, f.name))

    return ready

# ==========================================
# SUMMARY
# ==========================================
def generate_summary():
    print("\n" + "=" * 60)
    print("PODCAST AUDIO GENERATION SUMMARY")
    print("=" * 60)

    total_modules  = 0
    total_duration = 0.0
    total_turns    = 0
    total_failed   = 0
    chapter_stats  = {}

    for mc in sorted(AUDIO_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        ch_duration = 0.0
        ch_modules  = 0
        ch_turns    = 0
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
                    ch_turns += ameta.get("ok_count", 0)
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
                "turns_ok":           ch_turns,
                "turns_failed":       ch_failed,
                "duration_sec":       round(ch_duration, 1),
                "duration_formatted": f"{int(ch_duration//3600)}h {int((ch_duration%3600)//60)}m",
            }
            total_modules  += ch_modules
            total_duration += ch_duration
            total_turns    += ch_turns
            total_failed   += ch_failed

    print(f"\nTotal modules: {total_modules}")
    print(f"Total turns OK: {total_turns}")
    print(f"Total turns FAILED: {total_failed}")
    print(f"Total duration: {int(total_duration//3600)}h {int((total_duration%3600)//60)}m")
    if total_modules:
        print(f"Avg per module: {total_duration/total_modules:.0f}s ({total_duration/total_modules/60:.1f}min)")

    print(f"\nVoices: Teacher={TEACHER_VOICE} (male), Student={STUDENT_VOICE} (female)")
    print(f"Speaking rate: {SPEAKING_RATE}")

    print(f"\nPer chapter:")
    for code, info in sorted(chapter_stats.items()):
        fail_str = f", {info['turns_failed']} failed" if info['turns_failed'] else ""
        print(f"  {code}: {info['modules']} modules, {info['turns_ok']} turns{fail_str}, {info['duration_formatted']}")

    summary = {
        "total_modules":            total_modules,
        "total_turns_ok":           total_turns,
        "total_turns_failed":       total_failed,
        "total_duration_sec":       round(total_duration, 1),
        "total_duration_formatted": f"{int(total_duration//3600)}h {int((total_duration%3600)//60)}m",
        "teacher_voice":            TEACHER_VOICE,
        "student_voice":            STUDENT_VOICE,
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
    print(f"PODCAST AUDIO GENERATION")
    print(f"{'='*60}")
    print(f"  Model:           {MODEL}")
    print(f"  Speaking rate:   {SPEAKING_RATE}")
    print(f"  Teacher voice:   {TEACHER_VOICE} (male)")
    print(f"  Student voice:   {STUDENT_VOICE} (female)")
    print(f"  Podcasts from:   {PODCASTS_FOLDER}")
    print(f"  Audio output:    {AUDIO_FOLDER}")
    print(f"  Projects:        {len(cred_files)}")
    print(f"  RPM per project: {RPM_PER_PROJECT}")
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
        print(f"\nFound {len(ready)} modules ready for podcast audio (workers: {len(workers)})")

        # Round-robin across workers
        worker_assignments = [[] for _ in workers]
        for i, (mc, pf) in enumerate(ready):
            worker_assignments[i % len(workers)].append((mc, pf))

        def worker_run(w, modules):
            for mc, pf in modules:
                if w.dead:
                    break
                process_module(mc, pf, w)

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
