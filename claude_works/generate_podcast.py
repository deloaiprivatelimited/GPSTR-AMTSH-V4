"""
Generate two-person podcast scripts for each concept module.
- Reads module JSONs from claude_works/modules/{MERGE_CODE}/{MODULE_ID}.json
- Calls Gemini to generate podcast script JSON (teacher + student dialogue)
- Saves output to claude_works/podcasts/{MERGE_CODE}/{MODULE_ID}_podcast.json
- Multi-project parallel processing (same pattern as generate_chunks.py)
- Skips already-generated podcasts
- Retries on failure with exponential backoff
"""
import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.oauth2 import service_account

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
MODULES_FOLDER = Path("claude_works/modules")
PODCASTS_FOLDER = Path("claude_works/podcasts")
CREDENTIALS_FOLDER = Path("claude_works/credentials")
PROMPT_FILE = Path("prompts_v2/generate_audio_script.txt")

LOCATION = "us-central1"
MAX_WORKERS_PER_PROJECT = 4
DEBUG = False

PODCASTS_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────
# LOAD PROMPT
# ─────────────────────────────────────
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

# ─────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────
PODCAST_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "module_id": {"type": "STRING"},
        "module_title": {"type": "STRING"},
        "podcast_meta": {
            "type": "OBJECT",
            "properties": {
                "estimated_duration_minutes": {"type": "NUMBER"},
                "total_dialogues": {"type": "INTEGER"},
                "sections_covered": {"type": "ARRAY", "items": {"type": "STRING"}},
                "content_density": {"type": "STRING"}
            },
            "required": ["estimated_duration_minutes", "total_dialogues", "sections_covered", "content_density"]
        },
        "dialogues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "speaker": {"type": "STRING"},
                    "section": {"type": "STRING"},
                    "script": {"type": "STRING"}
                },
                "required": ["index", "speaker", "section", "script"]
            }
        }
    },
    "required": ["module_id", "module_title", "podcast_meta", "dialogues"]
}

# ─────────────────────────────────────
# MULTI-PROJECT INIT
# ─────────────────────────────────────
PROJECTS = []
for cred_file in sorted(os.listdir(CREDENTIALS_FOLDER)):
    if not cred_file.endswith(".json"):
        continue
    cred_path = os.path.join(CREDENTIALS_FOLDER, cred_file)
    creds = service_account.Credentials.from_service_account_file(cred_path)
    project_id = json.load(open(cred_path))["project_id"]
    PROJECTS.append({"project_id": project_id, "credentials": creds})
    print(f"  Loaded project: {project_id}")

print(f"Total projects: {len(PROJECTS)}")

_thread_local = threading.local()

def get_model(project_idx):
    key = f"model_{project_idx}"
    if not hasattr(_thread_local, key):
        p = PROJECTS[project_idx]
        vertexai.init(project=p["project_id"], location=LOCATION, credentials=p["credentials"])
        setattr(_thread_local, key, GenerativeModel("gemini-2.5-pro"))
    return getattr(_thread_local, key)

_thread_project_idx = threading.local()

# ─────────────────────────────────────
# TTS VALIDATION — catch bad characters
# ─────────────────────────────────────
BANNED_CHARS = set("()[]{}$\\^_=+*/%<>|~@#&≥≤≠→←∴∵≈²³√×÷π∞")
BANNED_WORDS = [
    "\\frac", "\\sqrt", "\\pm", "\\times", "\\div", "\\theta", "\\alpha", "\\beta",
    "$$", "LaTeX", "backslash", "caret", "underscore",
    "open bracket", "close bracket", "open parenthesis", "close parenthesis",
    "superscript", "subscript"
]

def validate_script(dialogues):
    """Check all script fields for banned chars/patterns. Returns list of warnings."""
    warnings = []
    for d in dialogues:
        script = d.get("script", "")
        idx = d.get("index", "?")
        # Check banned characters
        for ch in BANNED_CHARS:
            if ch in script:
                warnings.append(f"  Dialogue {idx}: banned char '{ch}' found")
        # Check banned words/patterns
        for word in BANNED_WORDS:
            if word in script:
                warnings.append(f"  Dialogue {idx}: banned pattern '{word}' found")
        # Check for $ (LaTeX marker)
        if "$" in script:
            warnings.append(f"  Dialogue {idx}: LaTeX $ marker found")
    return warnings

# ─────────────────────────────────────
# GEMINI CALL WITH RETRY
# ─────────────────────────────────────
def call_gemini(prompt, project_idx):
    model = get_model(project_idx)
    response = model.generate_content(
        [prompt],
        generation_config=GenerationConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=PODCAST_RESPONSE_SCHEMA,
        ),
    )
    return json.loads(response.text)

def safe_call(prompt, label, project_idx):
    max_retries = 15
    for attempt in range(max_retries):
        try:
            return call_gemini(prompt, project_idx), None
        except json.JSONDecodeError:
            wait = 10 * (attempt + 1)
            print(f"      JSON parse error, retrying in {wait}s... ({attempt+1}/{max_retries}) [{label}]")
            time.sleep(wait)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"      429 rate limit, waiting {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                time.sleep(wait)
            elif any(code in err for code in ["500", "502", "503", "504", "INTERNAL", "UNAVAILABLE"]):
                wait = 30 * (attempt + 1)
                print(f"      Server error, waiting {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                time.sleep(wait)
            elif "DEADLINE_EXCEEDED" in err or "timeout" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"      Timeout, waiting {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                time.sleep(wait)
            else:
                if attempt < 5:
                    wait = 15 * (attempt + 1)
                    print(f"      Error: {err[:100]}, retrying in {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                    time.sleep(wait)
                else:
                    return None, f"{label}: {e}"
    return None, f"{label}: max retries ({max_retries}) exhausted"

# ─────────────────────────────────────
# PROCESS ONE MODULE
# ─────────────────────────────────────
def process_module(merge_code, module_file, project_idx):
    module_path = MODULES_FOLDER / merge_code / module_file
    module_id = module_file.replace(".json", "")

    # Output path
    out_dir = PODCASTS_FOLDER / merge_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{module_id}_podcast.json"

    # Skip if already done
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("dialogues") and not existing.get("errors"):
                print(f"  >> Skip (done): {module_id}")
                return "skipped"
        except Exception:
            pass

    # Load module JSON
    module_data = json.loads(module_path.read_text(encoding="utf-8"))

    # Build prompt
    prompt = SYSTEM_PROMPT + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    prompt += "INPUT MODULE JSON:\n"
    prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    prompt += json.dumps(module_data, indent=2, ensure_ascii=False)

    # Quick content stats for logging
    theory = module_data.get("theory", {})
    n_def = len(theory.get("definitions", []))
    n_form = len(theory.get("formulas", []))
    n_thm = len(theory.get("theorems", []))
    n_prop = len(theory.get("properties", []))
    n_ex = len(module_data.get("worked_examples", []))
    print(f"  Generating: {module_id} (D:{n_def} F:{n_form} T:{n_thm} P:{n_prop} E:{n_ex})")

    # Call Gemini
    result, err = safe_call(prompt, module_id, project_idx)

    if err:
        # Save error marker
        error_data = {"module_id": module_id, "errors": [err], "dialogues": []}
        out_path.write_text(json.dumps(error_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  FAIL {module_id}: {err[:80]}")
        return "error"

    # Validate TTS safety
    dialogues = result.get("dialogues", [])
    warnings = validate_script(dialogues)
    if warnings:
        result["tts_warnings"] = warnings
        print(f"  WARN {module_id}: {len(warnings)} TTS warnings")
        for w in warnings[:5]:
            print(f"    {w}")

    # Save
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    n_dialogues = len(dialogues)
    duration = result.get("podcast_meta", {}).get("estimated_duration_minutes", "?")
    density = result.get("podcast_meta", {}).get("content_density", "?")
    print(f"  OK {module_id}: {n_dialogues} dialogues, ~{duration} min, density={density}")
    return "ok"

# ─────────────────────────────────────
# WORKER
# ─────────────────────────────────────
def process_module_with_project(args):
    project_idx, merge_code, module_file = args
    _thread_project_idx.idx = project_idx
    return process_module(merge_code, module_file, project_idx)

def run_project_batch(project_idx, chapters):
    p = PROJECTS[project_idx]
    all_tasks = []
    for code in chapters:
        module_dir = MODULES_FOLDER / code
        if not module_dir.is_dir():
            continue
        module_files = sorted([f for f in os.listdir(module_dir) if f.endswith(".json")])
        for mf in module_files:
            all_tasks.append((project_idx, code, mf))

    print(f"\n[Project {project_idx}] {p['project_id']} — {len(chapters)} chapters, {len(all_tasks)} modules")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PROJECT) as executor:
        executor.map(process_module_with_project, all_tasks)

    print(f"\n[Project {project_idx}] DONE")

# ─────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────
def generate_summary():
    print("\n" + "=" * 60)
    print("PODCAST GENERATION SUMMARY")
    print("=" * 60)

    total_modules = 0
    total_dialogues = 0
    total_errors = 0
    total_tts_warnings = 0
    total_duration = 0.0
    chapter_stats = {}

    for mc in sorted(os.listdir(PODCASTS_FOLDER)):
        mc_path = PODCASTS_FOLDER / mc
        if not mc_path.is_dir():
            continue
        ch_modules = 0
        ch_dialogues = 0
        ch_errors = 0
        ch_warnings = 0
        ch_duration = 0.0
        for f in sorted(os.listdir(mc_path)):
            if not f.endswith("_podcast.json"):
                continue
            try:
                data = json.loads((mc_path / f).read_text(encoding="utf-8"))
                ch_modules += 1
                dialogues = data.get("dialogues", [])
                ch_dialogues += len(dialogues)
                ch_errors += len(data.get("errors", []))
                ch_warnings += len(data.get("tts_warnings", []))
                ch_duration += data.get("podcast_meta", {}).get("estimated_duration_minutes", 0)
            except Exception:
                continue
        if ch_modules:
            chapter_stats[mc] = {
                "modules": ch_modules,
                "dialogues": ch_dialogues,
                "errors": ch_errors,
                "tts_warnings": ch_warnings,
                "duration_min": round(ch_duration, 1)
            }
            total_modules += ch_modules
            total_dialogues += ch_dialogues
            total_errors += ch_errors
            total_tts_warnings += ch_warnings
            total_duration += ch_duration

    print(f"\nTotal chapters: {len(chapter_stats)}")
    print(f"Total modules: {total_modules}")
    print(f"Total dialogues: {total_dialogues}")
    print(f"Total duration: ~{total_duration:.1f} min ({total_duration/60:.1f} hours)")
    print(f"Avg dialogues/module: {total_dialogues/max(total_modules,1):.1f}")
    print(f"Avg duration/module: {total_duration/max(total_modules,1):.1f} min")
    print(f"Total errors: {total_errors}")
    print(f"Total TTS warnings: {total_tts_warnings}")

    print(f"\nPer chapter:")
    for code, info in sorted(chapter_stats.items()):
        extras = []
        if info["errors"]:
            extras.append(f"{info['errors']} err")
        if info["tts_warnings"]:
            extras.append(f"{info['tts_warnings']} tts_warn")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(f"  {code}: {info['modules']} modules, {info['dialogues']} dialogues, ~{info['duration_min']} min{extra_str}")

    summary = {
        "total_chapters": len(chapter_stats),
        "total_modules": total_modules,
        "total_dialogues": total_dialogues,
        "total_duration_minutes": round(total_duration, 1),
        "total_errors": total_errors,
        "total_tts_warnings": total_tts_warnings,
        "chapters": chapter_stats
    }
    summary_path = PODCASTS_FOLDER / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")

# ─────────────────────────────────────
# MAIN
# ─────────────────────────────────────
def main():
    merge_codes = sorted([d for d in os.listdir(MODULES_FOLDER) if (MODULES_FOLDER / d).is_dir()])

    if DEBUG:
        merge_codes = merge_codes[:1]
        mc = merge_codes[0]
        module_files = sorted([f for f in os.listdir(MODULES_FOLDER / mc) if f.endswith(".json")])
        print(f"{'='*60}")
        print(f"DEBUG MODE — {mc} ({len(module_files)} modules)")
        print(f"{'='*60}")
        _thread_project_idx.idx = 0
        for mf in module_files:
            process_module(mc, mf, 0)
        generate_summary()
        print("\nDone")
        return

    total_modules = sum(
        len([f for f in os.listdir(MODULES_FOLDER / mc) if f.endswith(".json")])
        for mc in merge_codes
    )
    n_projects = len(PROJECTS)

    print(f"{'='*60}")
    print(f"PODCAST GENERATION — {n_projects} projects")
    print(f"{'='*60}")
    print(f"{len(merge_codes)} chapters, {total_modules} modules\n")

    # Split chapters across projects
    batches = [[] for _ in range(n_projects)]
    for i, code in enumerate(merge_codes):
        batches[i % n_projects].append(code)

    for i, batch in enumerate(batches):
        print(f"  Project {i} ({PROJECTS[i]['project_id']}): {len(batch)} chapters — {batch}")

    # Run each project batch in parallel
    with ThreadPoolExecutor(max_workers=n_projects) as executor:
        futures = []
        for i, batch in enumerate(batches):
            if batch:
                futures.append(executor.submit(run_project_batch, i, batch))
        for f in futures:
            f.result()

    generate_summary()
    print("\nDone")

if __name__ == "__main__":
    main()
