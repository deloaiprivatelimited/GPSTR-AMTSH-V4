"""
Check all audio modules for missing chunk WAVs.
Deletes entire audio folder for any concept with missing chunks.
Re-run generate_audio_multi.py after this to rebuild them cleanly.

Usage:
  python claude_works/fix_incomplete_audio.py          # dry run (just show)
  python claude_works/fix_incomplete_audio.py --delete  # actually delete
"""
import os
import json
import shutil
import sys

CHUNKS_FOLDER = "claude_works/chunks"
AUDIO_FOLDER = "claude_works/audio"

dry_run = "--delete" not in sys.argv

if dry_run:
    print("DRY RUN — showing what would be deleted. Use --delete to actually delete.\n")

complete = 0
incomplete = 0
no_audio = 0
deleted_list = []

for mc in sorted(os.listdir(CHUNKS_FOLDER)):
    mc_path = os.path.join(CHUNKS_FOLDER, mc)
    if not os.path.isdir(mc_path):
        continue

    for mod in sorted(os.listdir(mc_path)):
        mod_path = os.path.join(mc_path, mod)
        if not os.path.isdir(mod_path):
            continue

        meta_path = os.path.join(mod_path, "_meta.json")
        if not os.path.exists(meta_path):
            continue

        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("errors"):
            continue  # skip error modules

        chunk_order = meta.get("chunk_order", [])
        audio_dir = os.path.join(AUDIO_FOLDER, mc, mod)

        if not os.path.isdir(audio_dir):
            no_audio += 1
            continue

        # Check each chunk has WAV
        expected = 0
        missing = []
        for fn in chunk_order:
            chunk_path = os.path.join(mod_path, fn)
            if not os.path.exists(chunk_path):
                continue
            cd = json.load(open(chunk_path, encoding="utf-8"))
            script = cd.get("script", "").strip()
            if not script:
                continue
            expected += 1
            wav_name = fn.replace(".json", ".wav")
            wav_path = os.path.join(audio_dir, wav_name)
            if not os.path.exists(wav_path):
                missing.append(wav_name)

        if missing:
            incomplete += 1
            # Count existing wavs
            existing_wavs = len([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
            print(f"  INCOMPLETE: {mc}/{mod} — {len(missing)}/{expected} missing ({existing_wavs} WAVs exist)")
            for m in missing[:5]:
                print(f"    missing: {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")

            if not dry_run:
                shutil.rmtree(audio_dir)
                deleted_list.append(f"{mc}/{mod}")
                print(f"    DELETED: {audio_dir}")
            print()
        else:
            complete += 1

print("=" * 60)
print(f"Complete: {complete}")
print(f"Incomplete: {incomplete}")
print(f"No audio yet: {no_audio}")

if dry_run and incomplete > 0:
    print(f"\nRun with --delete to delete {incomplete} incomplete audio folders:")
    print(f"  python claude_works/fix_incomplete_audio.py --delete")
elif deleted_list:
    print(f"\nDeleted {len(deleted_list)} incomplete audio folders.")
    print(f"Re-run: python claude_works/generate_audio_multi.py")
elif incomplete == 0 and no_audio == 0:
    print(f"\nALL AUDIO COMPLETE!")
