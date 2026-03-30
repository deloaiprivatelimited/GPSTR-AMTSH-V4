"""
Chunk Validation + Fix Loop
- Per chunk file: validate + fix in one call
- If not production ready: retry up to 5 times
- Polls for new chunks from generation
- Stops when no new chunks found
"""
import os
import json
import time
import vertexai
from vertexai.generative_models import GenerativeModel
from concurrent.futures import ThreadPoolExecutor
import asyncio

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"

MODULES_FOLDER = "claude_works/modules"
CHUNKS_FOLDER = "claude_works/chunks"
REPORT_FOLDER = "claude_works/chunk_validation"

MAX_WORKERS = 3
MAX_FIX_ATTEMPTS = 5
POLL_INTERVAL = 30
DEBUG = False

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -----------------------------
# VALIDATION + FIX SCHEMA
# -----------------------------
VALIDATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "production_ready": {"type": "BOOLEAN"},
        "issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "field": {"type": "STRING"},
                    "problem": {"type": "STRING"},
                    "fix_applied": {"type": "STRING"}
                },
                "required": ["field", "problem", "fix_applied"]
            }
        },
        "fixed_chunk": {
            "type": "OBJECT",
            "properties": {
                "chunk_id": {"type": "STRING"},
                "type": {"type": "STRING"},
                "slide_title": {"type": "STRING"},
                "script": {"type": "STRING"},
                "script_display": {"type": "STRING"},
                "display_bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
                "layout_config": {
                    "type": "OBJECT",
                    "properties": {
                        "layout": {"type": "STRING"},
                        "template": {"type": "STRING"},
                        "text_zone": {"type": "STRING"},
                        "visual_zone": {"type": "STRING"},
                        "transition": {
                            "type": "OBJECT",
                            "properties": {
                                "in": {"type": "STRING"},
                                "out": {"type": "STRING"},
                                "duration_ms": {"type": "INTEGER"}
                            },
                            "required": ["in", "out", "duration_ms"]
                        }
                    },
                    "required": ["layout", "template", "text_zone", "visual_zone", "transition"]
                },
                "tts": {
                    "type": "OBJECT",
                    "properties": {
                        "read_field": {"type": "STRING"},
                        "language": {"type": "STRING"},
                        "sync_mode": {"type": "STRING"}
                    },
                    "required": ["read_field", "language", "sync_mode"]
                },
                "visual": {
                    "type": "OBJECT",
                    "properties": {
                        "type": {"type": "STRING"}
                    },
                    "required": ["type"]
                }
            },
            "required": ["type", "slide_title", "script", "display_bullets", "layout_config", "tts", "visual"]
        }
    },
    "required": ["production_ready", "issues", "fixed_chunk"]
}

# -----------------------------
# VALIDATION PROMPT
# -----------------------------
VALIDATE_AND_FIX_PROMPT = """
You are a PRODUCTION QA + FIXER for a GPSTR Kannada math video lesson slide.
This slide will become a video with TTS audio. ANY issue = bad video.

You get TWO inputs:
1. CHUNK JSON (the slide to validate)
2. SOURCE CONTENT (what it was generated from)

YOUR JOB: Validate AND fix in one shot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATE THESE (be STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TTS SAFETY (most critical — breaks audio):
   script field must have ZERO of these characters: ( ) [ ] { } $ \\ ^ _ = + - * / % < > | ~ @
   script must NOT contain: "sub", "superscript", "open bracket", "close bracket",
   "fraction", "over", "backslash", "LaTeX", "caret"
   No markdown: no **, no ##, no ```, no bullet markers like "- " or "* "
   No "..." continuation dots
   Math must be SPOKEN: "a squared" not "a^2", "a by b" not "a/b"

2. SCRIPT QUALITY:
   - Minimum 50 words for definition/formula/theorem slides
   - Minimum 40 words for worked example slides
   - Minimum 30 words for recap/answer slides
   - Must sound like a REAL Kannada math teacher — natural, warm, conversational
   - Must TEACH with concrete numerical examples (not just state facts)
   - For worked examples: must include ACTUAL numbers in calculations
     "substitute a equals 3 and d equals 5, we get 3 plus 9 times 5 which is 48"

3. SCRIPT-DISPLAY MATCH:
   - Whatever is in display_bullets MUST be mentioned in script
   - Student reads screen while hearing audio — they must match
   - display_bullets must be array of plain strings (not objects, not dicts)

4. MATHEMATICAL ACCURACY:
   - All formulas correct
   - All calculations correct
   - All theorem statements accurate
   - Compare against source content

5. STRUCTURE:
   - All required fields present: type, slide_title, script, display_bullets, layout_config, tts, visual
   - script_display should have LaTeX version of script
   - tts must be: {"read_field": "script", "language": "kn-IN", "sync_mode": "chunk"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return:
- production_ready: true if ALL checks pass, false if ANY fails
- issues: list of problems found (empty if production_ready)
- fixed_chunk: the CORRECTED version of the chunk
  If production_ready=true: return the chunk AS-IS (no changes)
  If production_ready=false: return the FIXED version with all issues resolved

FIXING RULES:
- Fix TTS issues: replace all symbols with spoken math
- Fix short scripts: expand with more detail and examples
- Fix display mismatch: make script cover all bullets
- Fix math errors: correct the formulas/calculations
- Keep same chunk_id, type, layout_config structure
- Keep everything that was correct — only fix what's broken
"""

REVALIDATE_PROMPT = """
You are doing a FINAL CHECK on a fixed video lesson slide.
This was previously NOT production ready and was fixed. Verify the fix worked.

PREVIOUS ISSUES:
{prev_issues}

FIXED CHUNK:
{chunk_json}

SOURCE CONTENT:
{source_json}

Check:
1. Are ALL previous issues actually fixed?
2. TTS safety: ZERO symbols in script? No ( ) [ ] { } $ ^ _ = + - %
3. Script quality: detailed enough? mentions display_bullets?
4. Math accuracy: correct?
5. No new issues introduced by the fix?

Return production_ready=true ONLY if everything is clean.
If still has issues: return production_ready=false with remaining issues and another fixed version.
"""

# ═════════════════════════════════════════
# GET SOURCE CONTENT
# ═════════════════════════════════════════

def get_source_content(module_data, chunk_filename):
    theory = module_data.get("theory", {})

    if "intro" in chunk_filename:
        return {"type": "intro", "concept_summary": theory.get("concept_summary", "")}

    if "definition" in chunk_filename:
        try:
            idx = int(chunk_filename.split("definition_")[1].split(".")[0].split("_")[0])
            defs = theory.get("definitions", [])
            if idx < len(defs):
                return {"type": "definition", "content": defs[idx]}
        except (ValueError, IndexError):
            pass
        return {"type": "definition", "content": theory.get("definitions", [])}

    if "theorem" in chunk_filename:
        try:
            idx = int(chunk_filename.split("theorem_")[1].split(".")[0].split("_")[0])
            thms = theory.get("theorems", [])
            if idx < len(thms):
                return {"type": "theorem", "content": thms[idx]}
        except (ValueError, IndexError):
            pass
        return {"type": "theorem", "content": theory.get("theorems", [])}

    if "property" in chunk_filename:
        return {"type": "property", "content": theory.get("properties", [])}

    if "formula" in chunk_filename:
        try:
            idx = int(chunk_filename.split("formula_")[1].split(".")[0].split("_")[0])
            formulas = theory.get("formulas", [])
            if idx < len(formulas):
                return {"type": "formula", "content": formulas[idx]}
        except (ValueError, IndexError):
            pass
        return {"type": "formula", "content": theory.get("formulas", [])}

    if "example" in chunk_filename:
        try:
            idx = int(chunk_filename.split("example_")[1].split(".")[0].split("_")[0])
            examples = module_data.get("worked_examples", [])
            if idx < len(examples):
                return {"type": "worked_example", "content": examples[idx]}
        except (ValueError, IndexError):
            pass
        return {"type": "worked_example", "content": module_data.get("worked_examples", [])}

    if "recap" in chunk_filename:
        return {"type": "recap", "exam_intelligence": module_data.get("exam_intelligence", {})}

    return {"type": "unknown"}

# ═════════════════════════════════════════
# VALIDATE + FIX ONE CHUNK
# ═════════════════════════════════════════

def validate_and_fix(chunk_data, source_content, is_revalidation=False, prev_issues=None):
    """One API call: validate + return fixed version"""

    if is_revalidation:
        prompt = f"""You are doing a FINAL CHECK on a fixed video lesson slide.
This was previously NOT production ready and was fixed. Verify the fix worked.

PREVIOUS ISSUES:
{json.dumps(prev_issues, indent=2, ensure_ascii=False)}

FIXED CHUNK:
{json.dumps(chunk_data, indent=2, ensure_ascii=False)}

SOURCE CONTENT:
{json.dumps(source_content, indent=2, ensure_ascii=False)}

Check:
1. Are ALL previous issues actually fixed?
2. TTS safety: ZERO symbols in script? No brackets, dollar signs, caret, equals, plus, minus, percent
3. Script quality: detailed enough? mentions display_bullets?
4. Math accuracy: correct?
5. No new issues introduced by the fix?

Return production_ready=true ONLY if everything is clean.
If still has issues: return production_ready=false with remaining issues and another fixed version."""
    else:
        prompt = f"""CHUNK TO VALIDATE AND FIX:
{json.dumps(chunk_data, indent=2, ensure_ascii=False)}

SOURCE CONTENT:
{json.dumps(source_content, indent=2, ensure_ascii=False)}

{VALIDATE_AND_FIX_PROMPT}"""

    try:
        response = model.generate_content(
            [prompt],
            generation_config={"temperature": 0.1}
        )

        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        result = json.loads(text)
        return result, None

    except Exception as e:
        return None, str(e)

# ═════════════════════════════════════════
# PROCESS ONE CHUNK FILE
# ═════════════════════════════════════════

def group_chunk_files(chunk_files):
    """Group chunk files from same API call.
    008_example_0.json + 009_example_cont_0_1.json + 010_example_cont_0_2.json → one group
    """
    groups = {}
    for cf in chunk_files:
        parts = cf.split("_", 1)
        if len(parts) < 2:
            groups[cf] = [cf]
            continue

        remainder = parts[1]

        if "_cont_" in remainder:
            parent_type = remainder.split("_cont_")[0]
            cont_part = remainder.split("_cont_")[1]
            parent_idx = cont_part.split("_")[0]
            group_key = f"{parent_type}_{parent_idx}"
        else:
            group_key = remainder.replace(".json", "")

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(cf)

    return groups

def process_chunk_group(group_key, group_files, chunk_dir, module_data, report_dir):
    """Validate + fix loop for a group of chunk files from same API call"""

    report_path = os.path.join(report_dir, f"{group_key}_report.json")

    if os.path.exists(report_path):
        return "skipped"

    # Load all chunks in this group
    all_chunks = []
    all_paths = []
    for cf in sorted(group_files):
        chunk_path = os.path.join(chunk_dir, cf)
        with open(chunk_path, "r", encoding="utf-8") as f:
            all_chunks.append(json.load(f))
        all_paths.append(chunk_path)

    # Get source content from first file
    source_content = get_source_content(module_data, group_files[0])
    theory = module_data.get("theory", {})
    source_content["module_context"] = {
        "module_id": module_data.get("module_id", ""),
        "module_title": module_data.get("module_title", ""),
        "chapter_title": module_data.get("chapter_title", ""),
        "class": module_data.get("class", ""),
        "domain": module_data.get("domain", ""),
        "concept_summary": theory.get("concept_summary", ""),
        "exam_intelligence": module_data.get("exam_intelligence", {}),
        "visual_aids": theory.get("visual_aids", [])
    }

    # Send as single chunk or array
    chunk_data = all_chunks if len(all_chunks) > 1 else all_chunks[0]

    # --- VALIDATION + FIX LOOP ---
    attempts = []
    current_chunk = chunk_data

    for attempt in range(MAX_FIX_ATTEMPTS):
        is_reval = attempt > 0
        prev_issues = attempts[-1]["issues"] if attempts else None

        result, err = validate_and_fix(current_chunk, source_content, is_reval, prev_issues)

        if err:
            attempts.append({"attempt": attempt + 1, "error": err})
            print(f"      API error on attempt {attempt+1}: {err[:80]}")
            time.sleep(3)
            continue

        production_ready = result.get("production_ready", False)
        issues = result.get("issues", [])
        fixed_chunk = result.get("fixed_chunk", current_chunk)

        attempts.append({
            "attempt": attempt + 1,
            "production_ready": production_ready,
            "issues_count": len(issues),
            "issues": issues
        })

        if production_ready:
            # Save fixed chunks back
            if isinstance(fixed_chunk, list):
                for idx, fc in enumerate(fixed_chunk):
                    if idx < len(all_paths):
                        with open(all_paths[idx], "w", encoding="utf-8") as f:
                            json.dump(fc, f, indent=2, ensure_ascii=False)
            else:
                with open(all_paths[0], "w", encoding="utf-8") as f:
                    json.dump(fixed_chunk, f, indent=2, ensure_ascii=False)
            break
        else:
            current_chunk = fixed_chunk
            # Save intermediate fix
            if isinstance(fixed_chunk, list):
                for idx, fc in enumerate(fixed_chunk):
                    if idx < len(all_paths):
                        with open(all_paths[idx], "w", encoding="utf-8") as f:
                            json.dump(fc, f, indent=2, ensure_ascii=False)
            else:
                with open(all_paths[0], "w", encoding="utf-8") as f:
                    json.dump(fixed_chunk, f, indent=2, ensure_ascii=False)

            print(f"      Attempt {attempt+1}: {len(issues)} issues, fixing...")

    # --- SAVE REPORT ---
    final_ready = attempts[-1].get("production_ready", False) if attempts else False

    report = {
        "group_key": group_key,
        "chunk_files": group_files,
        "production_ready": final_ready,
        "total_attempts": len(attempts),
        "attempts": attempts
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return "ready" if final_ready else "not_ready"

# ═════════════════════════════════════════
# PROCESS ONE MODULE
# ═════════════════════════════════════════

def process_module(merge_code, module_id):

    module_path = os.path.join(MODULES_FOLDER, merge_code, f"{module_id}.json")
    chunk_dir = os.path.join(CHUNKS_FOLDER, merge_code, module_id)
    meta_path = os.path.join(chunk_dir, "_meta.json")
    report_dir = os.path.join(REPORT_FOLDER, merge_code, module_id)

    if not os.path.exists(meta_path) or not os.path.exists(module_path):
        return

    os.makedirs(report_dir, exist_ok=True)

    # Check if already fully validated
    done_marker = os.path.join(report_dir, "_done.marker")
    if os.path.exists(done_marker):
        return

    with open(module_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)

    chunk_files = sorted([
        f for f in os.listdir(chunk_dir)
        if f.endswith(".json") and f != "_meta.json"
    ])

    if not chunk_files:
        return

    # Group by API call (example_0 + example_cont_0_1 → one group)
    groups = group_chunk_files(chunk_files)

    print(f"  Validating: {module_id} ({len(chunk_files)} chunks in {len(groups)} groups)")

    ready_count = 0
    not_ready_count = 0

    for group_key, group_files in sorted(groups.items()):
        print(f"    {group_key} ({len(group_files)} slides)...", end=" ")

        status = process_chunk_group(group_key, group_files, chunk_dir, module_data, report_dir)

        if status == "ready":
            print("READY")
            ready_count += 1
        elif status == "not_ready":
            print("NOT_READY")
            not_ready_count += 1
        else:
            print("skipped")
            ready_count += 1

    # Mark module as done
    with open(done_marker, "w") as f:
        f.write(f"ready: {ready_count}, not_ready: {not_ready_count}")

    verdict = "PRODUCTION_READY" if not_ready_count == 0 else f"NOT_READY ({not_ready_count} chunks)"
    print(f"  Result: {module_id}: {verdict}")

# ═════════════════════════════════════════
# FIND UNVALIDATED MODULES
# ═════════════════════════════════════════

def get_unvalidated_modules():
    unvalidated = []
    if not os.path.isdir(CHUNKS_FOLDER):
        return unvalidated

    for mc in sorted(os.listdir(CHUNKS_FOLDER)):
        mc_path = os.path.join(CHUNKS_FOLDER, mc)
        if not os.path.isdir(mc_path):
            continue
        for mod_dir in sorted(os.listdir(mc_path)):
            mod_path = os.path.join(mc_path, mod_dir)
            if not os.path.isdir(mod_path):
                continue
            meta_path = os.path.join(mod_path, "_meta.json")
            done_marker = os.path.join(REPORT_FOLDER, mc, mod_dir, "_done.marker")
            if os.path.exists(meta_path) and not os.path.exists(done_marker):
                unvalidated.append((mc, mod_dir))

    return unvalidated

# ═════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════

def generate_summary():
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total_modules = 0
    ready_modules = 0
    total_chunks = 0
    ready_chunks = 0
    not_ready_chunks = 0

    for mc in sorted(os.listdir(REPORT_FOLDER)):
        mc_path = os.path.join(REPORT_FOLDER, mc)
        if not os.path.isdir(mc_path):
            continue
        for mod_dir in sorted(os.listdir(mc_path)):
            mod_path = os.path.join(mc_path, mod_dir)
            if not os.path.isdir(mod_path):
                continue
            done_marker = os.path.join(mod_path, "_done.marker")
            if not os.path.exists(done_marker):
                continue

            total_modules += 1
            module_ready = True

            reports = [f for f in os.listdir(mod_path) if f.endswith("_report.json")]
            for rf in reports:
                try:
                    r = json.load(open(os.path.join(mod_path, rf), encoding="utf-8"))
                    total_chunks += 1
                    if r.get("production_ready"):
                        ready_chunks += 1
                    else:
                        not_ready_chunks += 1
                        module_ready = False
                except Exception:
                    pass

            if module_ready:
                ready_modules += 1

    print(f"\nModules: {ready_modules}/{total_modules} PRODUCTION_READY")
    print(f"Chunks: {ready_chunks}/{total_chunks} PRODUCTION_READY")
    if not_ready_chunks:
        print(f"Not ready: {not_ready_chunks} chunks")

    summary = {
        "total_modules": total_modules,
        "ready_modules": ready_modules,
        "total_chunks": total_chunks,
        "ready_chunks": ready_chunks,
        "not_ready_chunks": not_ready_chunks
    }

    summary_path = os.path.join(REPORT_FOLDER, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved: {summary_path}")

# ═════════════════════════════════════════
# MAIN — POLL MODE
# ═════════════════════════════════════════

async def main():
    print(f"{'='*60}")
    print(f"CHUNK VALIDATION + AUTO-FIX")
    print(f"{'='*60}")
    print(f"  Model: {MODEL}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Max fix attempts: {MAX_FIX_ATTEMPTS}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print()

    loop = asyncio.get_running_loop()
    total_processed = 0
    empty_polls = 0

    while True:
        unvalidated = get_unvalidated_modules()

        if DEBUG:
            unvalidated = unvalidated[:1]

        if not unvalidated:
            empty_polls += 1
            if empty_polls >= 2:
                print("No new chunks for 2 polls. Finishing.")
                break
            print(f"No new chunks. Waiting {POLL_INTERVAL}s... ({empty_polls}/2)")
            time.sleep(POLL_INTERVAL)
            continue

        empty_polls = 0
        print(f"\nFound {len(unvalidated)} modules to validate")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [
                loop.run_in_executor(executor, process_module, mc, mod_id)
                for mc, mod_id in unvalidated
            ]
            await asyncio.gather(*tasks)

        total_processed += len(unvalidated)
        print(f"\nTotal processed: {total_processed} modules")

    generate_summary()
    print("\nDone")

if __name__ == "__main__":
    asyncio.run(main())
