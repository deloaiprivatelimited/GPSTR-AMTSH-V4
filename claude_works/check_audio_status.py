"""
Audio generation status checker.
Shows: total chunks, completed, remaining, estimated time, avg chunk duration.
"""
import json
from pathlib import Path

CHUNKS_FOLDER = Path("claude_works/chunks")
AUDIO_FOLDER  = Path("claude_works/audio")
VALIDATION_FOLDER = Path("claude_works/chunk_validation")

def main():
    total_chunks = 0
    done_chunks = 0
    total_modules = 0
    done_modules = 0
    total_audio_duration = 0.0
    module_stats = []

    # Scan all chunk modules
    if not CHUNKS_FOLDER.is_dir():
        print("No chunks folder found!")
        return

    for mc in sorted(CHUNKS_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        for mod_dir in sorted(mc.iterdir()):
            if not mod_dir.is_dir():
                continue

            meta_path = mod_dir / "_meta.json"
            if not meta_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            chunk_order = meta.get("chunk_order", [])
            if not chunk_order:
                continue

            merge_code = mc.name
            module_id = mod_dir.name
            num_chunks = len(chunk_order)
            total_modules += 1
            total_chunks += num_chunks

            # Check audio status
            audio_dir = AUDIO_FOLDER / merge_code / module_id
            timeline_path = audio_dir / "timeline.json"

            if timeline_path.exists():
                done_modules += 1
                done_chunks += num_chunks
                try:
                    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
                    dur = tl.get("total_duration", 0)
                    total_audio_duration += dur
                    module_stats.append({
                        "module": f"{merge_code}/{module_id}",
                        "chunks": num_chunks,
                        "duration": dur,
                        "status": "done"
                    })
                except Exception:
                    pass
            else:
                # Count individual wav files done
                wavs_done = 0
                if audio_dir.is_dir():
                    for fn in chunk_order:
                        wav = audio_dir / fn.replace(".json", ".wav")
                        if wav.exists():
                            wavs_done += 1
                done_chunks += wavs_done
                module_stats.append({
                    "module": f"{merge_code}/{module_id}",
                    "chunks": num_chunks,
                    "wavs_done": wavs_done,
                    "status": "partial" if wavs_done > 0 else "pending"
                })

    remaining_chunks = total_chunks - done_chunks
    remaining_modules = total_modules - done_modules

    # Avg chunk audio duration (from completed modules)
    avg_chunk_dur = 0
    if done_chunks > 0 and total_audio_duration > 0:
        completed_chunk_count = sum(
            s["chunks"] for s in module_stats if s["status"] == "done"
        )
        if completed_chunk_count > 0:
            avg_chunk_dur = total_audio_duration / completed_chunk_count

    # ETA based on RPM
    rpm = 25  # effective RPM with 5 projects
    if remaining_chunks > 0:
        eta_minutes = remaining_chunks / rpm
        eta_h = int(eta_minutes // 60)
        eta_m = int(eta_minutes % 60)
    else:
        eta_h = 0
        eta_m = 0

    # Print report
    print("=" * 60)
    print("AUDIO GENERATION STATUS")
    print("=" * 60)

    print(f"\n  MODULES")
    print(f"    Total:      {total_modules}")
    print(f"    Done:       {done_modules}")
    print(f"    Remaining:  {remaining_modules}")
    pct_mod = (done_modules / total_modules * 100) if total_modules else 0
    print(f"    Progress:   {pct_mod:.1f}%")

    print(f"\n  CHUNKS")
    print(f"    Total:      {total_chunks}")
    print(f"    Done:       {done_chunks}")
    print(f"    Remaining:  {remaining_chunks}")
    pct_chunk = (done_chunks / total_chunks * 100) if total_chunks else 0
    print(f"    Progress:   {pct_chunk:.1f}%")

    print(f"\n  AUDIO OUTPUT")
    hrs = int(total_audio_duration // 3600)
    mins = int((total_audio_duration % 3600) // 60)
    secs = int(total_audio_duration % 60)
    print(f"    Generated:  {hrs}h {mins}m {secs}s of audio")
    print(f"    Avg chunk:  {avg_chunk_dur:.1f}s")

    print(f"\n  ETA (at {rpm} RPM)")
    print(f"    Remaining:  ~{eta_h}h {eta_m}m")

    # Per-chapter breakdown
    print(f"\n  PER CHAPTER:")
    chapter_data = {}
    for s in module_stats:
        ch = s["module"].split("/")[0]
        if ch not in chapter_data:
            chapter_data[ch] = {"total": 0, "done": 0, "chunks": 0, "chunks_done": 0, "duration": 0}
        chapter_data[ch]["total"] += 1
        chapter_data[ch]["chunks"] += s["chunks"]
        if s["status"] == "done":
            chapter_data[ch]["done"] += 1
            chapter_data[ch]["chunks_done"] += s["chunks"]
            chapter_data[ch]["duration"] += s.get("duration", 0)
        elif s.get("wavs_done", 0) > 0:
            chapter_data[ch]["chunks_done"] += s["wavs_done"]

    for ch, d in sorted(chapter_data.items()):
        dur_m = int(d["duration"] // 60)
        dur_s = int(d["duration"] % 60)
        pct = (d["chunks_done"] / d["chunks"] * 100) if d["chunks"] else 0
        print(f"    {ch:20s}  modules: {d['done']}/{d['total']}  chunks: {d['chunks_done']}/{d['chunks']} ({pct:.0f}%)  audio: {dur_m}m{dur_s:02d}s")

    # In-progress modules
    partial = [s for s in module_stats if s["status"] == "partial"]
    if partial:
        print(f"\n  IN PROGRESS ({len(partial)} modules):")
        for s in partial[:10]:
            print(f"    {s['module']:40s}  {s['wavs_done']}/{s['chunks']} chunks")
        if len(partial) > 10:
            print(f"    ... and {len(partial)-10} more")

if __name__ == "__main__":
    main()
