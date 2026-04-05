"""
Audio Progress Report
- Total hours generated
- Pending chunks (no WAV yet)
- Per-module breakdown
"""
import json
import wave
from pathlib import Path

CHUNKS_DIR = Path("claude_works/chunks_structured")
AUDIO_DIR  = Path("claude_works/audio_v2")

total_duration = 0.0
total_chunks = 0
total_done = 0
total_failed = 0
total_pending = 0
total_no_script = 0
modules_complete = 0
modules_partial = 0
modules_no_audio = 0

print("=" * 70)
print("AUDIO PROGRESS REPORT")
print("=" * 70)

for mc in sorted(CHUNKS_DIR.iterdir()):
    if not mc.is_dir():
        continue
    merge_code = mc.name

    for mod_dir in sorted(mc.iterdir()):
        if not mod_dir.is_dir():
            continue
        module_id = mod_dir.name
        meta_path = mod_dir / "_meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("errors"):
            continue

        chunk_order = meta.get("chunk_order", [])
        if not chunk_order:
            continue

        mod_done = 0
        mod_pending = 0
        mod_failed = 0
        mod_no_script = 0
        mod_duration = 0.0

        audio_meta_path = AUDIO_DIR / merge_code / module_id / "_audio_meta.json"
        audio_meta = {}
        if audio_meta_path.exists():
            audio_meta = json.loads(audio_meta_path.read_text(encoding="utf-8"))

        for cf in chunk_order:
            chunk_path = mod_dir / cf
            if not chunk_path.exists():
                continue

            chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
            script = chunk_data.get("script", "").strip()
            if not script:
                mod_no_script += 1
                continue

            total_chunks += 1
            wav_path = AUDIO_DIR / merge_code / module_id / cf.replace(".json", ".wav")

            if wav_path.exists():
                try:
                    with wave.open(str(wav_path), "rb") as wf:
                        dur = wf.getnframes() / float(wf.getframerate())
                    mod_duration += dur
                    mod_done += 1
                except Exception:
                    mod_failed += 1
            else:
                # Check audio_meta for failed status
                am = audio_meta.get("chunks", {}).get(cf, {})
                if am.get("status") == "failed":
                    mod_failed += 1
                else:
                    mod_pending += 1

        total_done += mod_done
        total_failed += mod_failed
        total_pending += mod_pending
        total_no_script += mod_no_script
        total_duration += mod_duration

        if mod_pending == 0 and mod_failed == 0 and mod_done > 0:
            modules_complete += 1
        elif mod_done > 0:
            modules_partial += 1
        else:
            modules_no_audio += 1

        # Only print modules with issues
        if mod_failed > 0 or mod_pending > 0:
            print(f"  {module_id:40s}  done={mod_done}  pending={mod_pending}  failed={mod_failed}  {mod_duration/60:.1f}min")

hours = total_duration / 3600
mins = (total_duration % 3600) / 60

print()
print("=" * 70)
print(f"TOTAL AUDIO:     {hours:.1f} hours ({int(hours)}h {int(mins)}m {int(total_duration%60)}s)")
print(f"CHUNKS:          {total_done} done / {total_pending} pending / {total_failed} failed / {total_chunks} total")
print(f"MODULES:         {modules_complete} complete / {modules_partial} partial / {modules_no_audio} no audio")
print(f"SKIPPED:         {total_no_script} chunks with no script")
pct = (total_done / total_chunks * 100) if total_chunks else 0
print(f"PROGRESS:        {pct:.1f}%")
print("=" * 70)
