"""
Chunks Progress Report
- How many modules have chunks generated
- How many pending (no chunks yet)
- Errors in generation
- Schema validation quick check
"""
import json
from pathlib import Path

MODULES_DIR = Path("claude_works/modules")
CHUNKS_DIR  = Path("claude_works/chunks_structured")

REQUIRED_FIELDS = {"type", "slide_title", "script", "script_display", "display_bullets", "layout_config", "tts", "visual"}

total_modules = 0
total_done = 0
total_pending = 0
total_with_errors = 0
total_chunks = 0
total_schema_bad = 0

print("=" * 70)
print("CHUNKS GENERATION PROGRESS")
print("=" * 70)

for mc in sorted(MODULES_DIR.iterdir()):
    if not mc.is_dir():
        continue
    merge_code = mc.name

    module_files = sorted([f for f in mc.iterdir() if f.suffix == ".json"])
    mc_done = 0
    mc_pending = 0
    mc_errors = 0
    mc_chunks = 0
    mc_schema_bad = 0

    for mf in module_files:
        module_id = mf.stem
        total_modules += 1

        chunk_dir = CHUNKS_DIR / merge_code / module_id
        meta_path = chunk_dir / "_meta.json"

        if not meta_path.exists():
            mc_pending += 1
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        errors = meta.get("errors", [])
        chunk_order = meta.get("chunk_order", [])

        if errors:
            mc_errors += 1
            print(f"  ERROR {module_id}: {len(errors)} errors - {errors[0][:80]}")
            continue

        mc_done += 1
        mc_chunks += len(chunk_order)

        # Quick schema check on first and last chunk
        for cf in [chunk_order[0], chunk_order[-1]] if len(chunk_order) >= 2 else chunk_order:
            cp = chunk_dir / cf
            if not cp.exists():
                continue
            chunk = json.loads(cp.read_text(encoding="utf-8"))
            missing = REQUIRED_FIELDS - set(chunk.keys())
            bad_fields = {"tts_config", "visual_aid", "slide_type"} & set(chunk.keys())
            if missing or bad_fields:
                mc_schema_bad += 1

    total_done += mc_done
    total_pending += mc_pending
    total_with_errors += mc_errors
    total_chunks += mc_chunks
    total_schema_bad += mc_schema_bad

    status = "OK" if mc_pending == 0 and mc_errors == 0 else ""
    if mc_pending > 0:
        status = f"PENDING={mc_pending}"
    if mc_errors > 0:
        status += f" ERRORS={mc_errors}"

    print(f"  {merge_code:20s}  modules={len(module_files):3d}  done={mc_done}  pending={mc_pending}  errors={mc_errors}  chunks={mc_chunks}  {status}")

print()
print("=" * 70)
pct = (total_done / total_modules * 100) if total_modules else 0
print(f"MODULES:       {total_done} done / {total_pending} pending / {total_with_errors} errors / {total_modules} total")
print(f"CHUNKS:        {total_chunks} total generated")
print(f"SCHEMA:        {total_schema_bad} chunks with wrong schema (spot check)")
print(f"PROGRESS:      {pct:.1f}%")
print("=" * 70)
