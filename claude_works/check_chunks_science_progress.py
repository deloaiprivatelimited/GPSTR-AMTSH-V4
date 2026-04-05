"""Check chunk generation progress for science modules."""
import os
import json
from pathlib import Path

MODULES_FOLDER = Path("claude_works/modules_science")
CHUNKS_FOLDER = Path("claude_works/chunks_science")

def main():
    if not MODULES_FOLDER.exists():
        print("modules_science/ not found")
        return

    total_modules = 0
    done_modules = 0
    error_modules = 0
    pending_modules = 0
    total_slides = 0
    total_errors = 0
    chapter_rows = []

    for chapter_dir in sorted(os.listdir(MODULES_FOLDER)):
        chapter_path = MODULES_FOLDER / chapter_dir
        if not chapter_path.is_dir():
            continue

        module_files = [f for f in os.listdir(chapter_path) if f.endswith(".json") and f != "validation.json"]
        ch_total = len(module_files)
        ch_done = 0
        ch_errors = 0
        ch_pending = 0
        ch_slides = 0
        ch_call_errors = 0

        for mf in module_files:
            module_id = mf.replace(".json", "")
            meta_path = CHUNKS_FOLDER / chapter_dir / module_id / "_meta.json"

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    errs = meta.get("errors", [])
                    if errs:
                        ch_errors += 1
                        ch_call_errors += len(errs)
                    else:
                        ch_done += 1
                    ch_slides += meta.get("total_slides", 0)
                except Exception:
                    ch_errors += 1
            else:
                ch_pending += 1

        total_modules += ch_total
        done_modules += ch_done
        error_modules += ch_errors
        pending_modules += ch_pending
        total_slides += ch_slides
        total_errors += ch_call_errors

        status = "DONE" if ch_pending == 0 and ch_errors == 0 else "PARTIAL" if ch_done > 0 else "PENDING"
        chapter_rows.append((chapter_dir, ch_total, ch_done, ch_errors, ch_pending, ch_slides, status))

    # Print
    pct = (done_modules / total_modules * 100) if total_modules else 0
    print(f"{'='*70}")
    print(f"SCIENCE CHUNKS PROGRESS")
    print(f"{'='*70}")
    print(f"  Modules: {done_modules}/{total_modules} done ({pct:.0f}%)")
    print(f"  Pending: {pending_modules}")
    print(f"  Errors:  {error_modules} modules ({total_errors} failed calls)")
    print(f"  Slides:  {total_slides}")
    print()

    print(f"{'Chapter':<25} {'Total':>5} {'Done':>5} {'Err':>5} {'Pend':>5} {'Slides':>7}  Status")
    print(f"{'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*7}  {'-'*7}")
    for row in chapter_rows:
        name, t, d, e, p, s, st = row
        # print(f"{name:<25} {t:>5} {d:>5} {e:>5} {p:>5} {s:>7}  {st}")

    if error_modules:
        print(f"\nModules with errors:")
        for chapter_dir in sorted(os.listdir(MODULES_FOLDER)):
            chapter_path = MODULES_FOLDER / chapter_dir
            if not chapter_path.is_dir():
                continue
            for mf in sorted(os.listdir(chapter_path)):
                if not mf.endswith(".json") or mf == "validation.json":
                    continue
                module_id = mf.replace(".json", "")
                meta_path = CHUNKS_FOLDER / chapter_dir / module_id / "_meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        errs = meta.get("errors", [])
                        if errs:
                            print(f"  {chapter_dir}/{module_id}: {len(errs)} errors")
                            for e in errs[:2]:
                                print(f"    - {e[:80]}")
                    except Exception:
                        pass

if __name__ == "__main__":
    main()
