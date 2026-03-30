"""
Check chunk integrity — verifies:
1. Sequence numbers are continuous (001, 002, 003... no gaps)
2. All chunks in _meta.json chunk_order actually exist
3. Each chunk has required fields (type, script, slide_title, display_bullets)
4. No empty scripts

Usage:
  python claude_works/check_chunks_integrity.py
"""
import os
import json
import re
import sys

CHUNKS_FOLDER = "claude_works/chunks"

total_modules = 0
clean_modules = 0
problem_modules = 0
all_issues = []

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

        total_modules += 1
        meta = json.load(open(meta_path, encoding="utf-8"))

        if meta.get("errors"):
            continue  # skip error modules

        chunk_order = meta.get("chunk_order", [])
        issues = []

        # 1. Check all files in chunk_order exist
        for fn in chunk_order:
            fp = os.path.join(mod_path, fn)
            if not os.path.exists(fp):
                issues.append(f"MISSING FILE: {fn}")

        # 2. Check sequence numbers are continuous
        chunk_files = sorted([
            f for f in os.listdir(mod_path)
            if f.endswith(".json") and f != "_meta.json"
        ])

        seq_numbers = []
        for f in chunk_files:
            match = re.match(r"^(\d+)_", f)
            if match:
                seq_numbers.append(int(match.group(1)))

        seq_numbers.sort()
        if seq_numbers:
            expected = list(range(seq_numbers[0], seq_numbers[0] + len(seq_numbers)))
            if seq_numbers != expected:
                gaps = set(expected) - set(seq_numbers)
                if gaps:
                    issues.append(f"SEQUENCE GAPS: missing {sorted(gaps)}")
                dupes = [n for n in seq_numbers if seq_numbers.count(n) > 1]
                if dupes:
                    issues.append(f"DUPLICATE SEQ: {set(dupes)}")

        # 3. Check chunk_order matches actual files
        actual_set = set(chunk_files)
        order_set = set(chunk_order)
        extra_files = actual_set - order_set
        missing_from_disk = order_set - actual_set

        if extra_files:
            issues.append(f"FILES NOT IN META: {sorted(extra_files)}")
        if missing_from_disk:
            issues.append(f"IN META BUT MISSING: {sorted(missing_from_disk)}")

        # 4. Check each chunk has required fields
        for fn in chunk_files:
            fp = os.path.join(mod_path, fn)
            try:
                chunk = json.load(open(fp, encoding="utf-8"))
            except json.JSONDecodeError:
                issues.append(f"INVALID JSON: {fn}")
                continue

            # Required fields
            if not chunk.get("type"):
                issues.append(f"NO TYPE: {fn}")
            if not chunk.get("slide_title"):
                issues.append(f"NO SLIDE_TITLE: {fn}")
            if not chunk.get("script", "").strip():
                issues.append(f"EMPTY SCRIPT: {fn}")
            if not chunk.get("display_bullets"):
                issues.append(f"NO DISPLAY_BULLETS: {fn}")
            elif not isinstance(chunk["display_bullets"], list):
                issues.append(f"BULLETS NOT LIST: {fn} (got {type(chunk['display_bullets']).__name__})")
            elif chunk["display_bullets"] and not isinstance(chunk["display_bullets"][0], str):
                issues.append(f"BULLETS NOT STRINGS: {fn}")
            if not chunk.get("layout_config"):
                issues.append(f"NO LAYOUT_CONFIG: {fn}")
            if not chunk.get("tts"):
                issues.append(f"NO TTS CONFIG: {fn}")

        # 5. Check chunk types make sense
        types_found = set()
        for fn in chunk_files:
            fp = os.path.join(mod_path, fn)
            try:
                chunk = json.load(open(fp, encoding="utf-8"))
                types_found.add(chunk.get("type", "unknown"))
            except:
                pass

        if "intro" not in types_found and chunk_files:
            issues.append("NO INTRO CHUNK")
        if "recap" not in types_found and chunk_files:
            issues.append("NO RECAP CHUNK")

        if issues:
            problem_modules += 1
            all_issues.append((f"{mc}/{mod}", len(chunk_files), issues))
        else:
            clean_modules += 1

# Print results
print("=" * 60)
print("CHUNK INTEGRITY CHECK")
print("=" * 60)
print(f"\nTotal modules: {total_modules}")
print(f"Clean: {clean_modules}")
print(f"Problems: {problem_modules}")

if all_issues:
    print(f"\n{'='*60}")
    print(f"ISSUES ({problem_modules} modules):")
    print(f"{'='*60}")
    for path, count, issues in all_issues:
        print(f"\n  {path} ({count} chunks):")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")

    # Summary of issue types
    issue_counts = {}
    for _, _, issues in all_issues:
        for issue in issues:
            key = issue.split(":")[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1

    print(f"\n{'='*60}")
    print("ISSUE SUMMARY:")
    for key, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")
else:
    print("\nALL CHUNKS CLEAN!")
