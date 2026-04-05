"""
Extract all scripts into one single .txt file.
All chapters, all modules, all slides -- just the teacher script.
Output: claude_works/scripts/all_scripts.txt
"""
import json
from pathlib import Path

CHUNKS_DIR = Path("claude_works/chunks_structured")
OUTPUT_DIR = Path("claude_works/scripts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lines = []
total_words = 0

for mc in sorted(CHUNKS_DIR.iterdir()):
    if not mc.is_dir():
        continue
    merge_code = mc.name
    chapter_title = ""

    for mod_dir in sorted(mc.iterdir()):
        if not mod_dir.is_dir():
            continue
        meta_path = mod_dir / "_meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("errors"):
            continue

        chunk_order = meta.get("chunk_order", [])
        if not chunk_order:
            continue

        if not chapter_title:
            chapter_title = meta.get("chapter_title", merge_code)
            lines.append("")
            lines.append("=" * 70)
            lines.append(f"CHAPTER: {chapter_title} ({merge_code})")
            lines.append("=" * 70)

        module_title = meta.get("module_title", mod_dir.name)
        lines.append("")
        lines.append(f"--- {module_title} ---")
        lines.append("")

        for cf in chunk_order:
            chunk_path = mod_dir / cf
            if not chunk_path.exists():
                continue
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
            script = chunk.get("script", "").strip()
            if script:
                total_words += len(script.split())
                lines.append(script)
                lines.append("")

out_path = OUTPUT_DIR / "all_scripts.txt"
out_path.write_text("\n".join(lines), encoding="utf-8")

hours = total_words / 130 / 60
print(f"Done: {out_path}")
print(f"Total words: {total_words} (~{hours:.1f} hours)")
