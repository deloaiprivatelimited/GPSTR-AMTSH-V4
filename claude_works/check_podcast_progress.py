"""
Podcast Pipeline Progress Report
Shows detailed status of both stages:
  Stage 1: Script generation (generate_podcast.py)
  Stage 2: Audio generation (generate_podcast_audio.py)
"""
import json
import wave
from pathlib import Path

MODULES_DIR  = Path("claude_works/modules")
PODCASTS_DIR = Path("claude_works/podcasts")
AUDIO_DIR    = Path("claude_works/podcast_audio")

# ─────────────────────────────────────
# STAGE 1: Script Generation
# ─────────────────────────────────────
total_modules = 0          # total concept JSONs in modules/
scripts_done = 0           # podcast JSON exists with dialogues
scripts_error = 0          # podcast JSON exists but has errors
scripts_pending = 0        # no podcast JSON yet
scripts_tts_warn = 0       # scripts with TTS warnings

total_dialogues = 0
total_est_duration = 0.0
density_counts = {"light": 0, "medium": 0, "heavy": 0}
section_counts = {}

per_chapter_scripts = {}   # merge_code -> {total, done, error, pending, dialogues, duration}

# Scan modules dir for total concept count
for mc in sorted(MODULES_DIR.iterdir()):
    if not mc.is_dir():
        continue
    merge_code = mc.name
    module_files = sorted([f for f in mc.iterdir() if f.suffix == ".json"])
    if not module_files:
        continue

    ch = {"total": len(module_files), "done": 0, "error": 0, "pending": 0,
          "dialogues": 0, "duration": 0.0, "tts_warn": 0}

    for mf in module_files:
        total_modules += 1
        module_id = mf.stem
        podcast_path = PODCASTS_DIR / merge_code / f"{module_id}_podcast.json"

        if not podcast_path.exists():
            scripts_pending += 1
            ch["pending"] += 1
            continue

        try:
            data = json.loads(podcast_path.read_text(encoding="utf-8"))
        except Exception:
            scripts_error += 1
            ch["error"] += 1
            continue

        if data.get("errors"):
            scripts_error += 1
            ch["error"] += 1
            continue

        dialogues = data.get("dialogues", [])
        if not dialogues:
            scripts_error += 1
            ch["error"] += 1
            continue

        scripts_done += 1
        ch["done"] += 1

        n_dlg = len(dialogues)
        total_dialogues += n_dlg
        ch["dialogues"] += n_dlg

        meta = data.get("podcast_meta", {})
        est = meta.get("estimated_duration_minutes", 0)
        total_est_duration += est
        ch["duration"] += est

        density = meta.get("content_density", "")
        if density in density_counts:
            density_counts[density] += 1

        for s in meta.get("sections_covered", []):
            section_counts[s] = section_counts.get(s, 0) + 1

        if data.get("tts_warnings"):
            scripts_tts_warn += 1
            ch["tts_warn"] += 1

    per_chapter_scripts[merge_code] = ch

# ─────────────────────────────────────
# STAGE 2: Audio Generation
# ─────────────────────────────────────
audio_modules_complete = 0
audio_modules_partial = 0
audio_modules_pending = 0
audio_turns_done = 0
audio_turns_failed = 0
audio_turns_pending = 0
audio_total_duration = 0.0
audio_teacher_duration = 0.0
audio_student_duration = 0.0

per_chapter_audio = {}     # merge_code -> {total, complete, partial, pending, turns_ok, turns_fail, duration}

for mc in sorted(PODCASTS_DIR.iterdir()):
    if not mc.is_dir():
        continue
    merge_code = mc.name
    podcast_files = sorted([f for f in mc.iterdir() if f.name.endswith("_podcast.json")])
    if not podcast_files:
        continue

    ch = {"total": len(podcast_files), "complete": 0, "partial": 0, "pending": 0,
          "turns_ok": 0, "turns_fail": 0, "turns_pending": 0, "duration": 0.0}

    for pf in podcast_files:
        module_id = pf.name.replace("_podcast.json", "")
        audio_dir = AUDIO_DIR / merge_code / module_id
        audio_meta_path = audio_dir / "_audio_meta.json"
        timeline_path = audio_dir / "timeline.json"

        # Load podcast to count expected turns
        try:
            pdata = json.loads(pf.read_text(encoding="utf-8"))
            if not pdata.get("dialogues") or pdata.get("errors"):
                continue
            expected_turns = len([d for d in pdata["dialogues"] if d.get("script", "").strip()])
        except Exception:
            continue

        if not audio_meta_path.exists():
            audio_modules_pending += 1
            ch["pending"] += 1
            ch["turns_pending"] += expected_turns
            audio_turns_pending += expected_turns
            continue

        try:
            ameta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
        except Exception:
            audio_modules_pending += 1
            ch["pending"] += 1
            ch["turns_pending"] += expected_turns
            audio_turns_pending += expected_turns
            continue

        ok = ameta.get("ok_count", 0)
        fail = ameta.get("failed_count", 0)
        pending = expected_turns - ok - fail

        audio_turns_done += ok
        audio_turns_failed += fail
        audio_turns_pending += max(pending, 0)
        ch["turns_ok"] += ok
        ch["turns_fail"] += fail
        ch["turns_pending"] += max(pending, 0)

        # Check timeline for duration
        if timeline_path.exists():
            try:
                tl = json.loads(timeline_path.read_text(encoding="utf-8"))
                dur = tl.get("total_duration", 0)
                audio_total_duration += dur
                ch["duration"] += dur
                audio_teacher_duration += tl.get("teacher_duration", 0)
                audio_student_duration += tl.get("student_duration", 0)
                audio_modules_complete += 1
                ch["complete"] += 1
            except Exception:
                if fail > 0:
                    audio_modules_partial += 1
                    ch["partial"] += 1
                else:
                    audio_modules_partial += 1
                    ch["partial"] += 1
        elif fail > 0:
            audio_modules_partial += 1
            ch["partial"] += 1
        elif ok > 0:
            audio_modules_partial += 1
            ch["partial"] += 1
        else:
            audio_modules_pending += 1
            ch["pending"] += 1

    per_chapter_audio[merge_code] = ch

# ─────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────
print()
print("=" * 75)
print("  PODCAST PIPELINE — DETAILED PROGRESS REPORT")
print("=" * 75)

# ── STAGE 1 ──
print()
print("─" * 75)
print("  STAGE 1: SCRIPT GENERATION (generate_podcast.py)")
print("─" * 75)

script_pct = (scripts_done / total_modules * 100) if total_modules else 0
bar_len = 40
filled = int(bar_len * scripts_done / total_modules) if total_modules else 0
bar = "█" * filled + "░" * (bar_len - filled)
print(f"\n  [{bar}] {script_pct:.1f}%")
print(f"\n  Modules:    {scripts_done} done / {scripts_error} error / {scripts_pending} pending / {total_modules} total")
print(f"  Dialogues:  {total_dialogues} total (avg {total_dialogues/max(scripts_done,1):.0f} per module)")
print(f"  Est audio:  {total_est_duration:.0f} min ({total_est_duration/60:.1f} hours)")
print(f"  TTS warns:  {scripts_tts_warn} scripts with banned chars in script field")
print(f"  Density:    light={density_counts['light']}  medium={density_counts['medium']}  heavy={density_counts['heavy']}")

if section_counts:
    top = sorted(section_counts.items(), key=lambda x: -x[1])
    print(f"  Sections:   {', '.join(f'{k}={v}' for k,v in top)}")

print(f"\n  Per chapter:")
print(f"  {'Chapter':<16} {'Total':>5} {'Done':>5} {'Err':>4} {'Pend':>5} {'Dlgs':>5} {'~Min':>6} {'TTS⚠':>5}")
print(f"  {'─'*16} {'─'*5} {'─'*5} {'─'*4} {'─'*5} {'─'*5} {'─'*6} {'─'*5}")
for code, ch in sorted(per_chapter_scripts.items()):
    print(f"  {code:<16} {ch['total']:>5} {ch['done']:>5} {ch['error']:>4} {ch['pending']:>5} "
          f"{ch['dialogues']:>5} {ch['duration']:>6.0f} {ch['tts_warn']:>5}")

# ── STAGE 2 ──
print()
print("─" * 75)
print("  STAGE 2: AUDIO GENERATION (generate_podcast_audio.py)")
print("─" * 75)

audio_total_turns = audio_turns_done + audio_turns_failed + audio_turns_pending
audio_pct = (audio_turns_done / audio_total_turns * 100) if audio_total_turns else 0
filled2 = int(bar_len * audio_turns_done / audio_total_turns) if audio_total_turns else 0
bar2 = "█" * filled2 + "░" * (bar_len - filled2)
print(f"\n  [{bar2}] {audio_pct:.1f}%")

audio_h = audio_total_duration / 3600
audio_m = (audio_total_duration % 3600) / 60
print(f"\n  Modules:    {audio_modules_complete} complete / {audio_modules_partial} partial / {audio_modules_pending} pending")
print(f"  Turns:      {audio_turns_done} done / {audio_turns_failed} failed / {audio_turns_pending} pending / {audio_total_turns} total")
print(f"  Duration:   {int(audio_h)}h {int(audio_m)}m ({audio_total_duration:.0f}s)")
if audio_total_duration > 0:
    print(f"  Split:      teacher={audio_teacher_duration:.0f}s ({audio_teacher_duration/audio_total_duration*100:.0f}%)  "
          f"student={audio_student_duration:.0f}s ({audio_student_duration/audio_total_duration*100:.0f}%)")
print(f"  Avg/module: {audio_total_duration/max(audio_modules_complete,1):.0f}s ({audio_total_duration/max(audio_modules_complete,1)/60:.1f}min)")

if per_chapter_audio:
    print(f"\n  Per chapter:")
    print(f"  {'Chapter':<16} {'Total':>5} {'Done':>5} {'Part':>5} {'Pend':>5} {'T_OK':>5} {'T_Fail':>6} {'T_Pend':>6} {'Dur':>8}")
    print(f"  {'─'*16} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*8}")
    for code, ch in sorted(per_chapter_audio.items()):
        dur_str = f"{ch['duration']/60:.1f}m" if ch["duration"] else "—"
        print(f"  {code:<16} {ch['total']:>5} {ch['complete']:>5} {ch['partial']:>5} {ch['pending']:>5} "
              f"{ch['turns_ok']:>5} {ch['turns_fail']:>6} {ch['turns_pending']:>6} {dur_str:>8}")

# ── OVERALL ──
print()
print("─" * 75)
print("  OVERALL PIPELINE")
print("─" * 75)

overall_pct = 0
if total_modules > 0:
    # Weight: scripts=40%, audio=60%
    s1 = scripts_done / total_modules
    s2 = (audio_modules_complete / scripts_done) if scripts_done else 0
    overall_pct = (s1 * 40 + s2 * 60)

filled3 = int(bar_len * overall_pct / 100)
bar3 = "█" * filled3 + "░" * (bar_len - filled3)
print(f"\n  [{bar3}] {overall_pct:.1f}%")
print(f"\n  Scripts:  {script_pct:.1f}% complete")
print(f"  Audio:    {audio_pct:.1f}% complete")
print(f"  Modules:  {total_modules} total → {scripts_done} scripted → {audio_modules_complete} fully audio")

print()
print("=" * 75)
