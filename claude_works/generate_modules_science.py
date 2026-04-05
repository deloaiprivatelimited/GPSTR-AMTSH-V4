"""
Generate structured module JSONs from science master data.
- Watches master_data_science/ for new .txt files (polling)
- Sends master data + PDF to Gemini to split into concept modules
- Validates JSON schema + content completeness before saving
- Uses podcast-tts credential, 4 parallel threads
"""
import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
LOCATION = "us-central1"

CREDENTIALS_DIR = Path("claude_works/credentials")
USE_ONLY = ["podcast-tts-488313-4868c70c14ad.json"]

MASTER_DATA_FOLDER = Path("claude_works/master_data_science")
PDF_FOLDER = Path("science_textbooks/split")
OUTPUT_FOLDER = Path("claude_works/modules_science")

THREADS = 4
POLL_INTERVAL = 30
MAX_POLLS = 200
DEBUG = False

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────
# PROJECT INIT
# ─────────────────────────────────────
PROJECTS = []
for cred_file in sorted(os.listdir(CREDENTIALS_DIR)):
    if not cred_file.endswith(".json"):
        continue
    if USE_ONLY and cred_file not in USE_ONLY:
        continue
    cred_path = CREDENTIALS_DIR / cred_file
    creds = service_account.Credentials.from_service_account_file(str(cred_path))
    project_id = json.load(open(cred_path))["project_id"]
    PROJECTS.append({"project_id": project_id, "credentials": creds})
    print(f"  Loaded project: {project_id}")

print(f"Total projects: {len(PROJECTS)}")

_thread_local = threading.local()

def get_model(project_idx=0):
    key = f"model_{project_idx}"
    if not hasattr(_thread_local, key):
        p = PROJECTS[project_idx]
        vertexai.init(project=p["project_id"], location=LOCATION, credentials=p["credentials"])
        setattr(_thread_local, key, GenerativeModel("gemini-2.5-pro"))
    return getattr(_thread_local, key)

# ─────────────────────────────────────
# MODULE SPLIT PROMPT
# ─────────────────────────────────────
MODULE_SPLIT_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCIENCE MASTER DATA → MODULE JSON FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ನೀವು ಕನ್ನಡ ಮಾಧ್ಯಮ GPSTR ವಿಜ್ಞಾನ LMS module splitter ಆಗಿದ್ದೀರಿ.

INPUT:  ಒಂದು chapter ನ master data (structured text with [ಮೂಲ_ಅವಧಾರಣೆ] blocks)
        + ORIGINAL PDF for cross-reference
OUTPUT: JSON object — ಪ್ರತಿ [ಮೂಲ_ಅವಧಾರಣೆ] → ಒಂದು standalone module

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPLITTING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* ಪ್ರತಿ [ಮೂಲ_ಅವಧಾರಣೆ_{N}] → ಒಂದು separate JSON module
* concept_id master data ದಿಂದ ನೇರ copy
* ಒಂದು concept ನ content ಬೇರೆ module ಗೆ leak ಆಗಬಾರದು
* ಕ್ರಮ master data ಅನುಸರಿಸಬೇಕು
* prerequisites → ಅದೇ chapter ನ ಹಿಂದಿನ concept_id ಗಳ list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE JSON SCHEMA (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each module must follow this EXACT schema:

{
  "module_id": "",
  "module_title": "",
  "class": "",
  "chapter": "",
  "chapter_title": "",
  "domain": "",
  "prerequisites": [],

  "theory": {
    "concept_summary": "",
    "definitions": [
      {"term": "", "definition": ""}
    ],
    "formulas": [
      {
        "formula_name": "",
        "latex": "",
        "condition": "",
        "exception": "",
        "derived_from": "",
        "used_in": "",
        "derivation": []
      }
    ],
    "laws": [
      {
        "law_name": "",
        "statement": "",
        "formula": "",
        "discovered_by": "",
        "applications": [],
        "exceptions": "",
        "common_error": ""
      }
    ],
    "chemical_equations": [
      {
        "equation": "",
        "type": "",
        "conditions": "",
        "description": ""
      }
    ],
    "properties": [],
    "classifications": [
      {
        "title": "",
        "categories": [
          {"name": "", "description": "", "examples": []}
        ]
      }
    ],
    "experiments": [
      {
        "title": "",
        "materials": "",
        "procedure": "",
        "observation": "",
        "conclusion": ""
      }
    ],
    "visual_aids": [
      {
        "type": "diagram | table | graph | flowchart | circuit | structure",
        "description": "",
        "labels": "",
        "scientific_significance": "",
        "table_data": ""
      }
    ]
  },

  "worked_examples": [
    {
      "example_id": "",
      "difficulty": "basic | intermediate | advanced",
      "problem": "",
      "steps": [
        {
          "step": 1,
          "action": "",
          "justification": "",
          "calculation": ""
        }
      ],
      "result": "",
      "common_trap": ""
    }
  ],

  "exam_intelligence": {
    "gpstr_weightage": "high | medium | low",
    "mcq_note": "",
    "two_mark_note": "",
    "five_mark_note": "",
    "common_mistakes": [],
    "boundary_conditions": [],
    "connects_to": []
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACT FAITHFULLY from master data:
✔ [ವ್ಯಾಖ್ಯೆ]                      → theory.definitions + theory.concept_summary
✔ [ಸೂತ್ರಗಳು / Formulas]          → theory.formulas (LaTeX ನೇರ copy — do NOT change)
✔ [ನಿಯಮಗಳು / Laws]              → theory.laws
✔ [ರಾಸಾಯನಿಕ ಸಮೀಕರಣಗಳು]        → theory.chemical_equations
✔ [ಅಂಶಪಟ್ಟಿ / Properties]        → theory.properties
✔ [ವರ್ಗೀಕರಣ / Classification]    → theory.classifications
✔ [ಪ್ರಾಯೋಗಿಕ ಚಟುವಟಿಕೆ]          → theory.experiments
✔ [ಕ್ರಮಬದ್ಧ ಉದಾಹರಣೆ]            → worked_examples (ALL steps + ALL calculations)
✔ [ದೃಶ್ಯ_ಸಹಾಯ]                   → theory.visual_aids
✔ [ವಿಶ್ಲೇಷಣೆ]                    → exam_intelligence (mistakes, boundary conditions)
✔ [ಶೈಕ್ಷಣಿಕ_ದೃಷ್ಟಿಕೋನ]          → exam_intelligence (weightage, mcq/2mark/5mark)

ALSO CHECK PDF:
✔ Cross-check all formulas, chemical equations, and diagrams against the PDF
✔ If master data missed a diagram/table visible in the PDF → add to visual_aids
✔ If a chemical equation is unbalanced in master data → fix from PDF

CRITICAL:
* LaTeX formulas — character-perfect copy from master data
* Chemical equations must be BALANCED with correct states
* Every worked example step MUST have a calculation field
* Do NOT hallucinate — zero invented content
* concept_summary must be a paraphrase, NOT a direct copy of definitions
* ALL figures, tables, flowcharts from master data must appear in visual_aids
* Units must be correct (SI preferred)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output a JSON object with concept_ids as keys:

{
  "{CHAPTER_CODE}_concept_1": { ...module JSON... },
  "{CHAPTER_CODE}_concept_2": { ...module JSON... },
  ...
}

* Valid JSON — no trailing commas, no comments
* Return ONLY the JSON. No commentary before or after.
"""

# ─────────────────────────────────────
# SCHEMA VALIDATION
# ─────────────────────────────────────
REQUIRED_TOP_KEYS = {"module_id", "module_title", "class", "chapter", "chapter_title", "domain", "prerequisites", "theory", "worked_examples", "exam_intelligence"}
REQUIRED_THEORY_KEYS = {"concept_summary", "definitions"}
REQUIRED_EXAM_KEYS = {"gpstr_weightage", "common_mistakes"}

def validate_module(concept_id, module):
    """Validate a single module against the schema. Returns list of errors."""
    errors = []

    # --- Top-level keys ---
    missing_top = REQUIRED_TOP_KEYS - set(module.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")

    # --- module_id match ---
    if module.get("module_id") != concept_id:
        errors.append(f"module_id '{module.get('module_id')}' != key '{concept_id}'")

    # --- theory block ---
    theory = module.get("theory")
    if not isinstance(theory, dict):
        errors.append("theory is not a dict")
    else:
        missing_theory = REQUIRED_THEORY_KEYS - set(theory.keys())
        if missing_theory:
            errors.append(f"Missing theory keys: {missing_theory}")

        # concept_summary not empty
        if not theory.get("concept_summary", "").strip():
            errors.append("concept_summary is empty")

        # definitions structure
        defs = theory.get("definitions", [])
        if not isinstance(defs, list):
            errors.append("definitions is not a list")
        else:
            for i, d in enumerate(defs):
                if not isinstance(d, dict):
                    errors.append(f"definitions[{i}] is not a dict")
                elif not d.get("term") or not d.get("definition"):
                    errors.append(f"definitions[{i}] missing term or definition")

        # formulas structure
        for i, f in enumerate(theory.get("formulas", [])):
            if not isinstance(f, dict):
                errors.append(f"formulas[{i}] is not a dict")
            elif not f.get("formula_name"):
                errors.append(f"formulas[{i}] missing formula_name")

        # laws structure
        for i, law in enumerate(theory.get("laws", [])):
            if not isinstance(law, dict):
                errors.append(f"laws[{i}] is not a dict")
            elif not law.get("law_name") or not law.get("statement"):
                errors.append(f"laws[{i}] missing law_name or statement")

        # chemical_equations structure
        for i, eq in enumerate(theory.get("chemical_equations", [])):
            if not isinstance(eq, dict):
                errors.append(f"chemical_equations[{i}] is not a dict")
            elif not eq.get("equation"):
                errors.append(f"chemical_equations[{i}] missing equation")

        # experiments structure
        for i, exp in enumerate(theory.get("experiments", [])):
            if not isinstance(exp, dict):
                errors.append(f"experiments[{i}] is not a dict")
            elif not exp.get("title"):
                errors.append(f"experiments[{i}] missing title")

        # visual_aids structure
        for i, va in enumerate(theory.get("visual_aids", [])):
            if not isinstance(va, dict):
                errors.append(f"visual_aids[{i}] is not a dict")
            elif not va.get("type") or not va.get("description"):
                errors.append(f"visual_aids[{i}] missing type or description")

    # --- worked_examples ---
    examples = module.get("worked_examples", [])
    if not isinstance(examples, list):
        errors.append("worked_examples is not a list")
    else:
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict):
                errors.append(f"worked_examples[{i}] is not a dict")
                continue
            if not ex.get("example_id"):
                errors.append(f"worked_examples[{i}] missing example_id")
            if not ex.get("problem"):
                errors.append(f"worked_examples[{i}] missing problem")
            if ex.get("difficulty") not in ("basic", "intermediate", "advanced"):
                errors.append(f"worked_examples[{i}] invalid difficulty: '{ex.get('difficulty')}'")
            steps = ex.get("steps", [])
            if not isinstance(steps, list) or not steps:
                errors.append(f"worked_examples[{i}] has no steps")
            else:
                for j, step in enumerate(steps):
                    if not isinstance(step, dict):
                        errors.append(f"worked_examples[{i}].steps[{j}] is not a dict")
                    elif not step.get("action"):
                        errors.append(f"worked_examples[{i}].steps[{j}] missing action")

    # --- exam_intelligence ---
    exam = module.get("exam_intelligence")
    if not isinstance(exam, dict):
        errors.append("exam_intelligence is not a dict")
    else:
        missing_exam = REQUIRED_EXAM_KEYS - set(exam.keys())
        if missing_exam:
            errors.append(f"Missing exam_intelligence keys: {missing_exam}")
        if exam.get("gpstr_weightage") not in ("high", "medium", "low"):
            errors.append(f"Invalid gpstr_weightage: '{exam.get('gpstr_weightage')}'")

    return errors

def validate_completeness(chapter_code, master_text, modules):
    """Check that modules cover all concepts from master data."""
    warnings = []

    # Count concepts in master data
    import re
    md_concepts = len(re.findall(r'\[ಮೂಲ_ಅವಧಾರಣೆ_\d+\]', master_text))
    mod_count = len(modules)
    if mod_count != md_concepts and md_concepts > 0:
        warnings.append(f"Master data has {md_concepts} concepts but {mod_count} modules generated")

    # Count definitions in master data vs modules
    md_defs = len(re.findall(r'(?:ವ್ಯಾಖ್ಯೆ|Definition)', master_text, re.IGNORECASE))
    mod_defs = sum(len(m.get("theory", {}).get("definitions", [])) for m in modules.values())
    if mod_defs == 0 and md_defs > 0:
        warnings.append(f"Master data has definitions but modules have 0")

    # Count examples in master data vs modules
    md_examples = master_text.count("example_id:") + master_text.count("ಕ್ರಮಬದ್ಧ ಉದಾಹರಣೆ")
    mod_examples = sum(len(m.get("worked_examples", [])) for m in modules.values())
    if mod_examples == 0 and md_examples > 2:
        warnings.append(f"Master data has examples but modules have 0")

    return warnings

# ─────────────────────────────────────
# HELPERS
# ─────────────────────────────────────
def txt_to_pdf_path(txt_name):
    """Convert 'std_6_chapter_1' → PDF path."""
    parts = txt_name.split("_", 2)
    if len(parts) < 3:
        return None
    std_dir = f"{parts[0]}_{parts[1]}"
    chapter_file = f"{parts[2]}.pdf"
    pdf_path = PDF_FOLDER / std_dir / chapter_file
    return pdf_path if pdf_path.exists() else None

def parse_chapter_info(txt_name):
    parts = txt_name.split("_", 2)
    if len(parts) < 3:
        return None, None
    return parts[1], parts[2]

def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)

# ─────────────────────────────────────
# FIND PENDING
# ─────────────────────────────────────
def find_pending():
    pending = []
    if not MASTER_DATA_FOLDER.exists():
        return pending
    for f in sorted(os.listdir(MASTER_DATA_FOLDER)):
        if not f.endswith(".txt") or f.endswith("_ERROR.txt"):
            continue
        name = f.replace(".txt", "")
        module_dir = OUTPUT_FOLDER / name
        # Skip if module dir exists and has JSON files
        if module_dir.exists():
            jsons = [x for x in os.listdir(module_dir) if x.endswith(".json") and x != "validation.json"]
            if jsons:
                continue
        pending.append(name)
    return pending

# ─────────────────────────────────────
# PROCESS ONE CHAPTER
# ─────────────────────────────────────
def process_chapter(txt_name):
    project_id = PROJECTS[0]["project_id"]
    master_path = MASTER_DATA_FOLDER / f"{txt_name}.txt"
    module_dir = OUTPUT_FOLDER / txt_name

    # Skip if already done
    if module_dir.exists():
        jsons = [x for x in os.listdir(module_dir) if x.endswith(".json") and x != "validation.json"]
        if jsons:
            print(f"  [{project_id}] Skip (done): {txt_name} ({len(jsons)} modules)")
            return "skipped"

    std_num, chapter = parse_chapter_info(txt_name)
    pdf_path = txt_to_pdf_path(txt_name)

    print(f"  [{project_id}] Generating modules: {txt_name} (Class {std_num})")

    master_text = master_path.read_text(encoding="utf-8")

    # Quick stats from master data
    import re
    n_concepts = len(re.findall(r'\[ಮೂಲ_ಅವಧಾರಣೆ_\d+\]', master_text))
    print(f"    Master data: {n_concepts} concepts detected")

    # Build prompt parts
    prompt_parts = []

    if pdf_path:
        with open(pdf_path, "rb") as f:
            pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")
        prompt_parts.append(pdf_part)

    full_prompt = f"""CHAPTER_CODE: {txt_name}
CLASS: {std_num}
CHAPTER: {chapter}

--- MASTER DATA ---
{master_text}
--- END MASTER DATA ---

{"The original PDF is also provided. Cross-check formulas, chemical equations, diagrams against it." if pdf_path else "No PDF available. Use master data only."}

{MODULE_SPLIT_PROMPT}"""

    prompt_parts.append(full_prompt)

    # Call Gemini with retry
    max_retries = 10
    last_err = ""
    for attempt in range(max_retries):
        try:
            model = get_model(0)
            response = model.generate_content(
                prompt_parts,
                generation_config={"temperature": 0.1}
            )

            modules = parse_json_response(response.text)

            # Handle list vs dict response
            if isinstance(modules, list):
                modules = {
                    m.get("module_id", f"{txt_name}_concept_{i+1}"): m
                    for i, m in enumerate(modules)
                }

            if not isinstance(modules, dict) or not modules:
                raise ValueError(f"Expected dict of modules, got {type(modules).__name__} with {len(modules) if modules else 0} items")

            # ── SCHEMA VALIDATION ──
            all_errors = {}
            valid_modules = {}
            for concept_id, module_data in modules.items():
                errs = validate_module(concept_id, module_data)
                if errs:
                    all_errors[concept_id] = errs
                    print(f"    SCHEMA WARN {concept_id}: {len(errs)} issues")
                    for e in errs[:3]:
                        print(f"      - {e}")
                valid_modules[concept_id] = module_data

            # ── COMPLETENESS VALIDATION ──
            completeness_warnings = validate_completeness(txt_name, master_text, valid_modules)
            if completeness_warnings:
                print(f"    COMPLETENESS WARN {txt_name}:")
                for w in completeness_warnings:
                    print(f"      - {w}")

            # ── SAVE MODULES ──
            module_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0
            for concept_id, module_data in valid_modules.items():
                module_path = module_dir / f"{concept_id}.json"
                module_path.write_text(
                    json.dumps(module_data, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                saved_count += 1

            # ── SAVE VALIDATION REPORT ──
            total_defs = sum(len(m.get("theory", {}).get("definitions", [])) for m in valid_modules.values())
            total_formulas = sum(len(m.get("theory", {}).get("formulas", [])) for m in valid_modules.values())
            total_laws = sum(len(m.get("theory", {}).get("laws", [])) for m in valid_modules.values())
            total_chem_eq = sum(len(m.get("theory", {}).get("chemical_equations", [])) for m in valid_modules.values())
            total_examples = sum(len(m.get("worked_examples", [])) for m in valid_modules.values())
            total_experiments = sum(len(m.get("theory", {}).get("experiments", [])) for m in valid_modules.values())
            total_visual = sum(len(m.get("theory", {}).get("visual_aids", [])) for m in valid_modules.values())

            validation = {
                "chapter_code": txt_name,
                "class": std_num,
                "total_modules": saved_count,
                "total_definitions": total_defs,
                "total_formulas": total_formulas,
                "total_laws": total_laws,
                "total_chemical_equations": total_chem_eq,
                "total_worked_examples": total_examples,
                "total_experiments": total_experiments,
                "total_visual_aids": total_visual,
                "schema_errors": all_errors if all_errors else None,
                "completeness_warnings": completeness_warnings if completeness_warnings else None,
                "pdf_cross_checked": pdf_path is not None,
            }

            val_path = module_dir / "validation.json"
            val_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"  [{project_id}] OK {txt_name}: {saved_count} modules (D:{total_defs} F:{total_formulas} L:{total_laws} CE:{total_chem_eq} Ex:{total_examples} Exp:{total_experiments})")
            return "ok"

        except json.JSONDecodeError as e:
            # Save raw for debugging
            module_dir.mkdir(parents=True, exist_ok=True)
            raw_path = module_dir / f"{txt_name}_RAW.txt"
            raw_path.write_text(response.text, encoding="utf-8")
            last_err = f"JSON parse error: {e}"
            wait = 15 * (attempt + 1)
            print(f"  [{project_id}] JSON error on {txt_name}, retrying in {wait}s... ({attempt+1}/{max_retries})")
            time.sleep(wait)

        except Exception as e:
            last_err = str(e)
            if "429" in last_err or "RESOURCE_EXHAUSTED" in last_err:
                wait = 60 * (attempt + 1)
                print(f"  [{project_id}] 429 on {txt_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif any(code in last_err for code in ["500", "502", "503", "504", "INTERNAL", "UNAVAILABLE"]):
                wait = 30 * (attempt + 1)
                print(f"  [{project_id}] Server error on {txt_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "DEADLINE_EXCEEDED" in last_err or "timeout" in last_err.lower():
                wait = 30 * (attempt + 1)
                print(f"  [{project_id}] Timeout on {txt_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                if attempt < 5:
                    wait = 15 * (attempt + 1)
                    print(f"  [{project_id}] Error on {txt_name}: {last_err[:100]}, retrying in {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    break

    # All retries failed
    module_dir.mkdir(parents=True, exist_ok=True)
    error_path = module_dir / f"{txt_name}_ERROR.txt"
    error_path.write_text(f"Max retries exhausted: {last_err}", encoding="utf-8")
    print(f"  [{project_id}] FAILED: {txt_name}")
    return "error"

# ─────────────────────────────────────
# PROCESS BATCH
# ─────────────────────────────────────
def process_batch(pending):
    results = {"ok": 0, "error": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(process_chapter, name): name for name in pending}
        for future in as_completed(futures):
            name = futures[future]
            try:
                status = future.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                print(f"  Unexpected error on {name}: {e}")
                results["error"] += 1
    return results

# ─────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────
def print_summary():
    total_chapters = 0
    total_modules = 0
    total_defs = 0
    total_examples = 0
    total_errors = 0
    chapter_stats = {}

    for d in sorted(os.listdir(OUTPUT_FOLDER)):
        dir_path = OUTPUT_FOLDER / d
        if not dir_path.is_dir():
            continue
        total_chapters += 1
        jsons = [f for f in os.listdir(dir_path) if f.endswith(".json") and f != "validation.json"]
        error_files = [f for f in os.listdir(dir_path) if f.endswith("_ERROR.txt")]

        n_mods = len(jsons)
        total_modules += n_mods

        if error_files:
            total_errors += 1

        # Read validation if exists
        val_path = dir_path / "validation.json"
        if val_path.exists():
            try:
                val = json.loads(val_path.read_text(encoding="utf-8"))
                total_defs += val.get("total_definitions", 0)
                total_examples += val.get("total_worked_examples", 0)
                chapter_stats[d] = val
            except Exception:
                chapter_stats[d] = {"total_modules": n_mods}
        else:
            chapter_stats[d] = {"total_modules": n_mods}

    print(f"\n{'='*60}")
    print(f"MODULE GENERATION SCIENCE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Total chapters: {total_chapters}")
    print(f"  Total modules: {total_modules}")
    print(f"  Total definitions: {total_defs}")
    print(f"  Total worked examples: {total_examples}")
    print(f"  Errors: {total_errors}")

    print(f"\nPer chapter:")
    for code, info in sorted(chapter_stats.items()):
        n = info.get("total_modules", 0)
        d = info.get("total_definitions", "?")
        ex = info.get("total_worked_examples", "?")
        warns = info.get("schema_errors") or info.get("completeness_warnings")
        flag = " (WARN)" if warns else ""
        print(f"  {code}: {n} modules, D:{d} Ex:{ex}{flag}")

    # Save summary
    summary = {
        "total_chapters": total_chapters,
        "total_modules": total_modules,
        "total_definitions": total_defs,
        "total_worked_examples": total_examples,
        "total_errors": total_errors,
        "chapters": {k: {"modules": v.get("total_modules", 0)} for k, v in chapter_stats.items()}
    }
    summary_path = OUTPUT_FOLDER / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")

# ─────────────────────────────────────
# MAIN — POLLING LOOP
# ─────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"SCIENCE MODULE GENERATION")
    print(f"{'='*60}")
    print(f"Credential: {PROJECTS[0]['project_id']}")
    print(f"Threads: {THREADS}")
    print(f"Polling interval: {POLL_INTERVAL}s")
    print(f"Watching: {MASTER_DATA_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}\n")

    total_processed = 0
    empty_polls = 0

    while empty_polls < MAX_POLLS:
        pending = find_pending()

        if DEBUG:
            pending = pending[:1]

        if not pending:
            empty_polls += 1
            if empty_polls == 1:
                print(f"\nNo pending files. Polling every {POLL_INTERVAL}s for new master data...")
            elif empty_polls % 10 == 0:
                print(f"  Still waiting... ({empty_polls} polls, {empty_polls * POLL_INTERVAL}s elapsed)")
            time.sleep(POLL_INTERVAL)
            continue

        empty_polls = 0
        print(f"\nFound {len(pending)} pending chapters: {pending[:5]}{'...' if len(pending) > 5 else ''}")

        results = process_batch(pending)
        total_processed += results.get("ok", 0)

        print(f"\nBatch done — ok:{results.get('ok',0)} error:{results.get('error',0)} skipped:{results.get('skipped',0)}")
        print(f"Total processed so far: {total_processed}")

        if DEBUG:
            break

    print_summary()
    print("\nDone!")

if __name__ == "__main__":
    main()
