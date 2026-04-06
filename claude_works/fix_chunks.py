"""
fix_chunks.py
--------------
Fixes display issues in chunk JSON files:
1. Converts **bold** markdown to <strong>bold</strong> in display_bullets and script_display
2. Parses table data from latex/placeholder_note JSON strings into proper headers/rows
3. Cleans up placeholder_note LLM thinking dumps
"""

import json
import re
from pathlib import Path

CHUNKS_DIR = Path("claude_works/chunks_structured")

stats = {"files_scanned": 0, "bold_fixed": 0, "tables_fixed": 0, "notes_cleaned": 0}


def fix_bold_markdown(text):
    """Convert **text** to <strong>text</strong>"""
    if not text or "**" not in text:
        return text, False
    fixed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return fixed, fixed != text


def parse_table_from_latex(latex_str):
    """Try to parse table headers/rows from latex JSON string."""
    if not latex_str:
        return None, None
    try:
        data = json.loads(latex_str)
        if isinstance(data, list) and len(data) > 0:
            # Format: [{"header": "...", "content": [...]}, ...]
            if all(isinstance(d, dict) and "header" in d for d in data):
                headers = [d["header"] for d in data]
                max_rows = max(len(d.get("content", [])) for d in data)
                rows = []
                for i in range(max_rows):
                    row = []
                    for d in data:
                        content = d.get("content", [])
                        row.append(content[i] if i < len(content) else "")
                    rows.append(row)
                return headers, rows

            # Format: [["h1","h2"], ["r1c1","r1c2"], ...]
            if all(isinstance(d, list) for d in data):
                return data[0], data[1:]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None, None


def clean_placeholder_note(note):
    """Remove LLM thinking dumps from placeholder_note."""
    if not note:
        return note, False
    # If note is excessively long (>500 chars), it's likely a thinking dump
    if len(note) > 500:
        # Try to extract just the first sentence
        first_line = note.split(".")[0] + "." if "." in note[:200] else note[:200]
        return first_line.strip(), True
    return note, False


def fix_chunk_file(filepath):
    """Fix a single chunk JSON file. Returns True if modified."""
    with open(filepath, "r", encoding="utf-8") as f:
        chunk = json.load(f)

    modified = False

    # Fix **bold** in display_bullets
    bullets = chunk.get("display_bullets", [])
    new_bullets = []
    for b in bullets:
        fixed, changed = fix_bold_markdown(b)
        new_bullets.append(fixed)
        if changed:
            modified = True
            stats["bold_fixed"] += 1
    if modified:
        chunk["display_bullets"] = new_bullets

    # Fix **bold** in script_display
    sd = chunk.get("script_display", "")
    if sd:
        fixed_sd, changed = fix_bold_markdown(sd)
        if changed:
            chunk["script_display"] = fixed_sd
            modified = True
            stats["bold_fixed"] += 1

    # Fix tables: parse latex JSON string into headers/rows
    visual = chunk.get("visual", {})
    if visual.get("type") == "table":
        latex = visual.get("latex", "")
        if latex and not visual.get("headers"):
            headers, rows = parse_table_from_latex(latex)
            if headers and rows:
                visual["headers"] = headers
                visual["rows"] = rows
                modified = True
                stats["tables_fixed"] += 1

        # Clean placeholder_note
        note = visual.get("placeholder_note", "")
        cleaned, changed = clean_placeholder_note(note)
        if changed:
            visual["placeholder_note"] = cleaned
            modified = True
            stats["notes_cleaned"] += 1

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

    return modified


def main():
    print("Fixing chunks...\n")

    modified_count = 0
    all_files = list(CHUNKS_DIR.rglob("*.json"))
    all_files = [f for f in all_files if f.name != "_meta.json" and f.name != "generation_summary.json"]

    for filepath in all_files:
        stats["files_scanned"] += 1
        if fix_chunk_file(filepath):
            modified_count += 1

    print(f"Scanned:        {stats['files_scanned']} files")
    print(f"Modified:       {modified_count} files")
    print(f"Bold fixed:     {stats['bold_fixed']} occurrences")
    print(f"Tables fixed:   {stats['tables_fixed']} tables")
    print(f"Notes cleaned:  {stats['notes_cleaned']} notes")


if __name__ == "__main__":
    main()
