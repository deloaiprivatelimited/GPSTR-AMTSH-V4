import os
import json
import vertexai
from concurrent.futures import ThreadPoolExecutor
import asyncio
from vertexai.generative_models import GenerativeModel, Part

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"

PDF_FOLDER = "merged"
MODULES_FOLDER = "claude_works/modules"
VALIDATION_FOLDER = "claude_works/module_validation"
DATA_PATH = "data.json"

MAX_WORKERS = 3
DEBUG = False

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

# -----------------------------
# LOAD DATA MAP
# -----------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

code_info = {}
for domain, chapters in data.items():
    for ch in chapters:
        code = ch["merge_code"]
        if code not in code_info:
            code_info[code] = {"domain": domain, "chapters": []}
        code_info[code]["chapters"].append({
            "class": ch["class"],
            "chapter_no": ch["chapter_no"],
            "chapter_name": ch["chapter_name"]
        })

# -----------------------------
# FIX PROMPT
# -----------------------------
FIX_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX MODULES — ADD MISSING CONTENT FROM PDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE: You are fixing GPSTR/HSTR exam preparation modules.
You have THREE inputs:
1. The ORIGINAL PDF (source of truth)
2. The CURRENT MODULES (JSON — have gaps)
3. The VALIDATION REPORT (tells you exactly what's missing/wrong)

YOUR JOB: Return the COMPLETE FIXED set of all modules for this chapter.
Keep everything that's already correct. Only add/fix what the report says.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MISSING WORKED EXAMPLES — This is the #1 issue.
   - Find each missing example in the PDF
   - Extract it with COMPLETE step-by-step solution
   - Every step MUST have: action, justification, calculation
   - Place it in the correct module (matching concept)
   - Set appropriate difficulty (basic/intermediate/advanced)
   - NEVER write "similarly..." or skip steps

2. MISSING DEFINITIONS / PROPERTIES
   - Find in PDF, add to the correct module

3. MISSING FIGURES / VISUAL AIDS
   - Describe from PDF, add to correct module's visual_aids

4. WRONG FORMULAS / ANSWERS
   - Compare with PDF, fix the LaTeX or calculation

5. HALLUCINATED CONTENT
   - Remove anything not in the PDF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE JSON SCHEMA (same as before)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each module must have:
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
    "definitions": [{"term": "", "definition": ""}],
    "formulas": [{"formula_name": "", "latex": "", "condition": "", "exception": "", "derived_from": "", "used_in": "", "derivation": []}],
    "theorems": [{"theorem_name": "", "statement": "", "proof": "", "converse": "", "special_cases": [], "common_error": ""}],
    "properties": [],
    "visual_aids": [{"type": "", "description": "", "labels": "", "mathematical_significance": "", "table_data": ""}]
  },
  "worked_examples": [
    {
      "example_id": "",
      "difficulty": "basic | intermediate | advanced",
      "problem": "",
      "steps": [{"step": 1, "action": "", "justification": "", "calculation": ""}],
      "result": "",
      "common_trap": ""
    }
  ],
  "exam_intelligence": {
    "gpstr_weightage": "high | medium | low",
    "mcq_note": "", "two_mark_note": "", "five_mark_note": "",
    "common_mistakes": [], "boundary_conditions": [], "connects_to": []
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* Keep ALL existing correct content — do NOT remove or rewrite what's good
* Only add/fix what the validation report flags
* PDF is the source of truth for all additions
* Place new examples in the module whose concept they belong to
* If a missing example doesn't fit any existing module, create a new module
* LaTeX must be character-perfect from the PDF
* All explanations in ಕನ್ನಡ
* Math terms: ಕನ್ನಡ ಪದ (English term)
* Zero hallucination — every addition must come from the PDF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return a JSON object with concept_ids as keys — the COMPLETE fixed module set:

{
  "{MERGE_CODE}_concept_1": { ...full module JSON... },
  "{MERGE_CODE}_concept_2": { ...full module JSON... },
  ...
}

Return ALL modules (not just changed ones). This replaces the old set entirely.
Return ONLY valid JSON. No commentary.
"""

# -----------------------------
# LOAD FAILING CHAPTERS
# -----------------------------
def get_chapters_to_fix():
    summary_path = os.path.join(VALIDATION_FOLDER, "validation_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    to_fix = []

    for ch in summary.get("failed_chapters", []):
        to_fix.append(ch["merge_code"])

    for ch in summary.get("needs_fix_chapters", []):
        to_fix.append(ch["merge_code"])

    return sorted(set(to_fix))

# -----------------------------
# FIX ONE CHAPTER
# -----------------------------
def fix_chapter(merge_code):

    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    module_dir = os.path.join(MODULES_FOLDER, merge_code)
    validation_path = os.path.join(VALIDATION_FOLDER, f"{merge_code}_validation.json")
    fix_marker = os.path.join(module_dir, f"_FIXED.marker")

    # Skip if already fixed
    if os.path.exists(fix_marker):
        print(f"⏭ Already fixed: {merge_code}")
        return

    if not os.path.exists(pdf_path):
        print(f"⚠ No PDF: {merge_code}")
        return

    if not os.path.exists(validation_path):
        print(f"⚠ No validation report: {merge_code}")
        return

    # Load current modules
    module_files = sorted([f for f in os.listdir(module_dir) if f.endswith(".json")])
    all_modules = {}
    for mf in module_files:
        with open(os.path.join(module_dir, mf), "r", encoding="utf-8") as f:
            all_modules[os.path.splitext(mf)[0]] = json.load(f)

    # Load validation report
    with open(validation_path, "r", encoding="utf-8") as f:
        validation = json.load(f)

    # Build issue summary
    missing = validation.get("coverage", {}).get("missing_from_modules", [])
    accuracy = validation.get("accuracy", {})
    per_module_issues = []
    for pm in validation.get("per_module", []):
        if pm.get("verdict") == "HAS_ISSUES":
            per_module_issues.append(pm)

    print(f"🔧 Fixing: {merge_code} ({len(module_files)} modules, "
          f"{len(missing)} missing items)")

    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")

    modules_json = json.dumps(all_modules, indent=2, ensure_ascii=False)

    info = code_info.get(merge_code, {})
    chapters_str = ""
    if info:
        chapters_str = f"Domain: {info['domain']}\n"
        for ch in info["chapters"]:
            chapters_str += f"  - Class {ch['class']}, Ch {ch['chapter_no']}: {ch['chapter_name']}\n"

    # Build validation summary for the prompt
    validation_summary = f"""Overall verdict: {validation.get('overall_verdict', '?')}
Summary: {validation.get('summary', '')}

MISSING FROM MODULES (must add these):
{json.dumps(missing, indent=2, ensure_ascii=False)}

ACCURACY ISSUES:
  Wrong formulas: {json.dumps(accuracy.get('wrong_formulas', []), ensure_ascii=False)}
  Wrong answers: {json.dumps(accuracy.get('wrong_answers', []), ensure_ascii=False)}
  Hallucinated: {json.dumps(accuracy.get('hallucinated_content', []), ensure_ascii=False)}
  Incomplete examples: {json.dumps(accuracy.get('incomplete_examples', []), ensure_ascii=False)}

PER-MODULE ISSUES:
{json.dumps(per_module_issues, indent=2, ensure_ascii=False)}"""

    prompt = f"""MERGE_CODE: {merge_code}
{chapters_str}

--- CURRENT MODULES (JSON) ---
{modules_json}
--- END MODULES ---

--- VALIDATION REPORT ---
{validation_summary}
--- END VALIDATION REPORT ---

{FIX_PROMPT}"""

    try:
        response = model.generate_content(
            [pdf_part, prompt],
            generation_config={"temperature": 0.1}
        )

        result_text = response.text.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(lines)

        fixed_modules = json.loads(result_text)

        # Handle array response
        if isinstance(fixed_modules, list):
            fixed_dict = {}
            for m in fixed_modules:
                mid = m.get("module_id", f"{merge_code}_concept_{len(fixed_dict)+1}")
                fixed_dict[mid] = m
            fixed_modules = fixed_dict

        # Count changes
        old_examples = sum(
            len(m.get("worked_examples", []))
            for m in all_modules.values()
        )
        new_examples = sum(
            len(m.get("worked_examples", []))
            for m in fixed_modules.values()
        )

        # Remove old module files
        for mf in module_files:
            os.remove(os.path.join(module_dir, mf))

        # Save fixed modules
        for concept_id, module_data in fixed_modules.items():
            module_path = os.path.join(module_dir, f"{concept_id}.json")
            with open(module_path, "w", encoding="utf-8") as f:
                json.dump(module_data, f, indent=2, ensure_ascii=False)

        # Write fix marker
        with open(fix_marker, "w", encoding="utf-8") as f:
            f.write(f"Fixed: {len(fixed_modules)} modules, "
                    f"examples {old_examples} -> {new_examples}")

        print(f"  ✅ {merge_code}: {len(all_modules)} -> {len(fixed_modules)} modules, "
              f"examples {old_examples} -> {new_examples}")

    except json.JSONDecodeError:
        error_path = os.path.join(module_dir, f"{merge_code}_FIX_RAW.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  !! {merge_code}: JSON parse error - raw saved")

    except Exception as e:
        error_path = os.path.join(module_dir, f"{merge_code}_FIX_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"  XX {merge_code}: {e}")

# -----------------------------
# SUMMARY
# -----------------------------
def print_summary(chapters_to_fix):
    print("\n" + "=" * 60)
    print("📊 FIX SUMMARY")
    print("=" * 60)

    fixed = 0
    failed = 0

    for code in chapters_to_fix:
        module_dir = os.path.join(MODULES_FOLDER, code)
        fix_marker = os.path.join(module_dir, "_FIXED.marker")

        if os.path.exists(fix_marker):
            with open(fix_marker, "r", encoding="utf-8") as f:
                print(f"  OK {code}: {f.read()}")
            fixed += 1
        else:
            error_files = [f for f in os.listdir(module_dir)
                          if f.endswith("_FIX_ERROR.txt") or f.endswith("_FIX_RAW.txt")]
            if error_files:
                print(f"  ❌ {code}: Fix failed")
                failed += 1
            else:
                print(f"  ⏭ {code}: Not attempted")

    print(f"\nFixed: {fixed} / {len(chapters_to_fix)}")
    if failed:
        print(f"Failed: {failed}")

    print(f"\n→ Next: re-run validate_modules.py to confirm fixes")

# -----------------------------
# ASYNC RUNNER
# -----------------------------
async def main():

    chapters_to_fix = get_chapters_to_fix()

    if DEBUG:
        chapters_to_fix = chapters_to_fix[:1]

    print(f"{'='*60}")
    print(f"🔧 FIXING {len(chapters_to_fix)} CHAPTERS")
    print(f"{'='*60}")
    print(f"Chapters: {', '.join(chapters_to_fix)}\n")

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, fix_chapter, code)
            for code in chapters_to_fix
        ]
        await asyncio.gather(*tasks)

    print_summary(chapters_to_fix)

if __name__ == "__main__":
    asyncio.run(main())
