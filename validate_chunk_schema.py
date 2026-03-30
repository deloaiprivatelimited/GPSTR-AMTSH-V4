"""
Chunk Schema Validator
Scans all chunk JSON files and reports schema inconsistencies.
Produces a CSV report + console summary.
"""

import json
import os
import csv
from collections import defaultdict, Counter
from pathlib import Path

CHUNKS_DIR = Path("claude_works/chunks")
REPORT_CSV = Path("chunk_schema_report.csv")
SUMMARY_FILE = Path("chunk_schema_summary.txt")

# ---------------------------------------------------------------------------
# 1. Collect every JSON's top-level keys + nested structure
# ---------------------------------------------------------------------------

def flatten_keys(obj, prefix=""):
    """Return a set of dot-separated key paths for a nested dict."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.add(full)
            if isinstance(v, dict):
                keys |= flatten_keys(v, full)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                keys |= flatten_keys(v[0], f"{full}[]")
    return keys


def detect_field_variants(data):
    """Check for known field name variants and return issues."""
    issues = []

    # TTS field
    if "tts" in data and "tts_config" not in data:
        issues.append(("tts_field", "tts"))
    elif "tts_config" in data and "tts" not in data:
        issues.append(("tts_field", "tts_config"))
    elif "tts" in data and "tts_config" in data:
        issues.append(("tts_field", "BOTH tts & tts_config"))
    else:
        issues.append(("tts_field", "MISSING"))

    # Visual field
    if "visual" in data and "visual_aid" not in data:
        issues.append(("visual_field", "visual"))
    elif "visual_aid" in data and "visual" not in data:
        issues.append(("visual_field", "visual_aid"))
    elif "visual" in data and "visual_aid" in data:
        issues.append(("visual_field", "BOTH visual & visual_aid"))
    else:
        issues.append(("visual_field", "MISSING"))

    # Title field
    if "slide_title" in data:
        issues.append(("title_field", "slide_title"))
    elif "title" in data:
        issues.append(("title_field", "title"))
    else:
        issues.append(("title_field", "MISSING"))

    # Type field
    if "type" in data and "slide_type" not in data:
        issues.append(("type_field", "type"))
    elif "slide_type" in data and "type" not in data:
        issues.append(("type_field", "slide_type"))
    elif "type" in data and "slide_type" in data:
        issues.append(("type_field", "BOTH type & slide_type"))
    else:
        issues.append(("type_field", "MISSING"))

    # Script location
    if "script" in data:
        issues.append(("script_location", "top_level"))
    elif "content" in data and isinstance(data["content"], dict) and "script" in data["content"]:
        issues.append(("script_location", "inside_content"))
    else:
        issues.append(("script_location", "MISSING"))

    # script_display
    if "script_display" in data:
        issues.append(("has_script_display", "yes"))
    elif "content" in data and isinstance(data["content"], dict) and "script_display" in data.get("content", {}):
        issues.append(("has_script_display", "yes (in content)"))
    else:
        issues.append(("has_script_display", "no"))

    # display_bullets location
    if "display_bullets" in data:
        issues.append(("bullets_location", "top_level"))
    elif "content" in data and isinstance(data["content"], dict) and "steps" in data["content"]:
        issues.append(("bullets_location", "content.steps"))
    else:
        issues.append(("bullets_location", "MISSING"))

    # chunk_id
    if "chunk_id" in data:
        issues.append(("has_chunk_id", "yes"))
    else:
        issues.append(("has_chunk_id", "MISSING"))

    # Extra fields that only some schemas have
    for extra in ["slug", "module_id", "slide_id", "slide_number", "source_example_id", "notes"]:
        if extra in data:
            issues.append((f"extra_{extra}", "present"))

    return issues


# ---------------------------------------------------------------------------
# 2. Scan all files
# ---------------------------------------------------------------------------

results = []         # per-file records
field_counters = defaultdict(Counter)  # field_name -> {variant: count}
top_level_key_sets = Counter()         # frozenset of top-level keys -> count
all_key_paths = Counter()              # every dot-path -> count
module_schemas = defaultdict(list)     # module_dir -> list of top-level key frozensets
errors = []

total = 0
for root, dirs, files in os.walk(CHUNKS_DIR):
    for fname in sorted(files):
        if not fname.endswith(".json") or fname == "generation_summary.json":
            continue
        fpath = Path(root) / fname
        total += 1
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            errors.append((str(fpath), str(e)))
            continue

        rel = fpath.relative_to(CHUNKS_DIR)
        module_dir = str(rel.parts[0]) if rel.parts else "unknown"

        top_keys = frozenset(data.keys())
        top_level_key_sets[top_keys] += 1
        module_schemas[module_dir].append(top_keys)

        for kp in flatten_keys(data):
            all_key_paths[kp] += 1

        issues = detect_field_variants(data)
        row = {"file": str(rel), "module": module_dir, "top_level_keys": ",".join(sorted(data.keys()))}
        for field_name, variant in issues:
            row[field_name] = variant
            field_counters[field_name][variant] += 1

        results.append(row)

# ---------------------------------------------------------------------------
# 3. Write CSV report
# ---------------------------------------------------------------------------

all_columns = ["file", "module", "top_level_keys",
               "tts_field", "visual_field", "title_field", "type_field",
               "script_location", "has_script_display", "bullets_location",
               "has_chunk_id"]
# Add extra_* columns that appeared
extra_cols = sorted({k for r in results for k in r if k.startswith("extra_")})
all_columns += extra_cols

with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

# ---------------------------------------------------------------------------
# 4. Build summary
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 70)
lines.append("CHUNK SCHEMA VALIDATION SUMMARY")
lines.append("=" * 70)
lines.append(f"Total files scanned: {total}")
lines.append(f"Parse errors: {len(errors)}")
lines.append(f"Unique top-level key schemas: {len(top_level_key_sets)}")
lines.append("")

# Field variant breakdown
lines.append("-" * 70)
lines.append("FIELD VARIANT BREAKDOWN (key inconsistencies)")
lines.append("-" * 70)
for field_name in ["tts_field", "visual_field", "title_field", "type_field",
                    "script_location", "has_script_display", "bullets_location", "has_chunk_id"]:
    lines.append(f"\n  {field_name}:")
    for variant, count in field_counters[field_name].most_common():
        pct = count / total * 100
        flag = " <<<" if pct < 50 else ""
        lines.append(f"    {variant:30s} {count:5d}  ({pct:5.1f}%){flag}")

# Extra fields
lines.append(f"\n  Extra fields (present in some but not all):")
for col in extra_cols:
    count = sum(1 for r in results if col in r)
    pct = count / total * 100
    lines.append(f"    {col:30s} {count:5d}  ({pct:5.1f}%)")

# Per-module schema consistency
lines.append("")
lines.append("-" * 70)
lines.append("PER-MODULE SCHEMA CONSISTENCY")
lines.append("-" * 70)
inconsistent_modules = []
for mod in sorted(module_schemas):
    unique = len(set(module_schemas[mod]))
    total_in_mod = len(module_schemas[mod])
    if unique > 1:
        inconsistent_modules.append((mod, unique, total_in_mod))
        lines.append(f"  {mod:30s}  {unique} different schemas across {total_in_mod} files  *** INCONSISTENT ***")
    else:
        lines.append(f"  {mod:30s}  OK ({total_in_mod} files, 1 schema)")

lines.append("")
lines.append(f"Modules with internal inconsistencies: {len(inconsistent_modules)} / {len(module_schemas)}")

# Top schema patterns
lines.append("")
lines.append("-" * 70)
lines.append(f"TOP 10 MOST COMMON SCHEMAS (out of {len(top_level_key_sets)} unique)")
lines.append("-" * 70)
for i, (keyset, count) in enumerate(top_level_key_sets.most_common(10), 1):
    lines.append(f"\n  Schema #{i} ({count} files):")
    lines.append(f"    Keys: {', '.join(sorted(keyset))}")

# Parse errors
if errors:
    lines.append("")
    lines.append("-" * 70)
    lines.append("PARSE ERRORS")
    lines.append("-" * 70)
    for fpath, err in errors:
        lines.append(f"  {fpath}: {err}")

lines.append("")
lines.append("=" * 70)
lines.append(f"Detailed per-file report: {REPORT_CSV}")
lines.append("=" * 70)

summary = "\n".join(lines)

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
