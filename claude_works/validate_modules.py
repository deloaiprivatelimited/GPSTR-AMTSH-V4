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
DATA_PATH = "data.json"
OUTPUT_FOLDER = "claude_works/module_validation"

MAX_WORKERS = 3
DEBUG = False

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
# VALIDATION PROMPT
# -----------------------------
VALIDATE_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE VALIDATION — ALL MODULES vs SOURCE PDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE: You are a Mathematics content auditor for GPSTR/HSTR exam preparation.
You have TWO inputs:
1. The ORIGINAL PDF (source of truth)
2. ALL MODULE JSONs for this topic (combined)

Your job: verify that the modules TOGETHER cover everything in the PDF
correctly, completely, and without errors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECK 1: TOPIC-LEVEL COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Go through the PDF section by section and verify that across ALL modules combined:

- [ ] Every section/subsection in the PDF is covered by some module
- [ ] Every definition in the PDF exists in some module
- [ ] Every formula in the PDF exists in some module with correct LaTeX
- [ ] Every theorem in the PDF exists in some module (with proof if PDF has it)
- [ ] Every worked example in the PDF exists in some module with ALL steps
- [ ] Every figure/diagram in the PDF is described in some module's visual_aids
- [ ] Every table in the PDF is captured in some module
- [ ] Every property/rule in the PDF is in some module's properties
- [ ] No content from the PDF is completely missing across all modules

List anything from the PDF that is NOT covered by any module.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECK 2: MATHEMATICAL ACCURACY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each module, verify against the PDF:

- [ ] All formula LaTeX matches the PDF exactly
- [ ] Theorem statements are faithful to the PDF
- [ ] Worked example solutions are correct (verify final answers)
- [ ] Every example step has action + calculation (no empty calculations)
- [ ] Proof steps are complete and logically valid
- [ ] No hallucinated content (anything NOT in the PDF)
- [ ] Definitions match the textbook

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECK 3: MODULE QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per module:
- [ ] concept_summary is meaningful (not just repeating definitions)
- [ ] Concept boundaries make sense (not too big, not too small)
- [ ] No content duplicated across modules
- [ ] Prerequisites correctly reference prior concepts
- [ ] exam_intelligence has specific GPSTR-relevant tips (not generic)
- [ ] Difficulty tags on examples make sense
- [ ] visual_aids have proper descriptions and labels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECK 4: STRUCTURAL VALIDITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] All required JSON fields present (module_id, theory, worked_examples, exam_intelligence)
- [ ] module_id follows {MERGE_CODE}_concept_{N} pattern
- [ ] LaTeX syntax is valid (balanced $, correct commands)
- [ ] No empty arrays where content should exist
- [ ] Modules ordered foundational → advanced

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO IGNORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do NOT flag these as issues:
✗ Unsolved exercise questions from PDF (intentionally excluded)
✗ Chapter introductions / learning outcomes
✗ Activity sections (unless they derive a formula)
✗ "Did You Know" boxes
✗ Minor wording differences in explanations (as long as meaning is correct)
✗ Slight reordering within a concept (as long as all content present)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — Return ONLY valid JSON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "merge_code": "",
  "total_modules": 0,
  "overall_verdict": "PASS" | "NEEDS_FIX" | "CRITICAL_FAIL",
  "confidence": 0.0-1.0,
  "summary": "One-line overall assessment",

  "coverage": {
    "verdict": "complete" | "partial" | "major_gaps",
    "definitions_in_pdf": 0,
    "definitions_in_modules": 0,
    "formulas_in_pdf": 0,
    "formulas_in_modules": 0,
    "theorems_in_pdf": 0,
    "theorems_in_modules": 0,
    "examples_in_pdf": 0,
    "examples_in_modules": 0,
    "figures_in_pdf": 0,
    "figures_in_modules": 0,
    "missing_from_modules": [
      "List specific items from PDF not covered by any module"
    ]
  },

  "accuracy": {
    "verdict": "correct" | "minor_errors" | "critical_errors",
    "wrong_formulas": [],
    "wrong_answers": [],
    "hallucinated_content": [],
    "incomplete_examples": []
  },

  "per_module": [
    {
      "module_id": "",
      "module_title": "",
      "verdict": "GOOD" | "HAS_ISSUES",
      "issues": [
        {
          "severity": "critical" | "major",
          "type": "missing_content" | "wrong_formula" | "wrong_answer" | "incomplete_example" | "hallucination" | "structural",
          "description": "",
          "expected": "",
          "actual": ""
        }
      ]
    }
  ],

  "duplication": {
    "found": true | false,
    "details": []
  }
}

SCORING:
- "PASS" = coverage complete, accuracy correct, no critical/major issues
- "NEEDS_FIX" = minor gaps or errors that should be fixed
- "CRITICAL_FAIL" = major content missing, wrong formulas, wrong answers
- Only report critical and major issues. Skip minor/cosmetic things.
- Return ONLY the JSON.
"""

# -----------------------------
# VALIDATE ONE MERGE CODE
# -----------------------------
def validate_chapter(merge_code):

    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    module_dir = os.path.join(MODULES_FOLDER, merge_code)
    output_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_validation.json")

    if os.path.exists(output_path):
        print(f"⏭ Skipping: {merge_code}")
        return

    if not os.path.exists(pdf_path):
        print(f"⚠ No PDF: {merge_code}")
        return

    if not os.path.exists(module_dir):
        print(f"⚠ No modules: {merge_code}")
        return

    # Load all module JSONs
    module_files = sorted([f for f in os.listdir(module_dir) if f.endswith(".json")])
    if not module_files:
        print(f"⚠ No JSON modules in: {merge_code}")
        return

    all_modules = {}
    for mf in module_files:
        with open(os.path.join(module_dir, mf), "r", encoding="utf-8") as f:
            module_data = json.load(f)
        concept_id = os.path.splitext(mf)[0]
        all_modules[concept_id] = module_data

    modules_json = json.dumps(all_modules, indent=2, ensure_ascii=False)

    print(f"🔍 Validating: {merge_code} ({len(module_files)} modules)")

    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")

    info = code_info.get(merge_code, {})
    chapters_str = ""
    if info:
        chapters_str = f"Domain: {info['domain']}\n"
        for ch in info["chapters"]:
            chapters_str += f"  - Class {ch['class']}, Ch {ch['chapter_no']}: {ch['chapter_name']}\n"

    prompt = f"""MERGE_CODE: {merge_code}
{chapters_str}
Total modules: {len(module_files)}
Module IDs: {', '.join(os.path.splitext(f)[0] for f in module_files)}

--- ALL MODULES (JSON) ---
{modules_json}
--- END MODULES ---

{VALIDATE_PROMPT}"""

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

        result = json.loads(result_text)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        verdict = result.get("overall_verdict", "?")
        coverage = result.get("coverage", {}).get("verdict", "?")
        accuracy = result.get("accuracy", {}).get("verdict", "?")
        missing = len(result.get("coverage", {}).get("missing_from_modules", []))

        print(f"  ✅ {merge_code}: {verdict} | "
              f"coverage={coverage} accuracy={accuracy} | "
              f"{missing} missing items")

    except json.JSONDecodeError:
        error_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_RAW.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  ⚠ {merge_code}: JSON parse error — raw saved")

    except Exception as e:
        error_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"  ❌ {merge_code}: {e}")

# -----------------------------
# GENERATE SUMMARY
# -----------------------------
def generate_summary():
    print("\n" + "=" * 60)
    print("📊 MODULE VALIDATION SUMMARY")
    print("=" * 60)

    results = []
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith("_validation.json"):
            with open(os.path.join(OUTPUT_FOLDER, f), "r", encoding="utf-8") as fh:
                results.append(json.load(fh))

    if not results:
        print("No results found.")
        return

    total = len(results)
    pass_count = sum(1 for r in results if r.get("overall_verdict") == "PASS")
    fix_count = sum(1 for r in results if r.get("overall_verdict") == "NEEDS_FIX")
    fail_count = sum(1 for r in results if r.get("overall_verdict") == "CRITICAL_FAIL")

    print(f"\nTotal chapters validated: {total}")
    print(f"  ✅ PASS:          {pass_count}")
    print(f"  ⚠  NEEDS_FIX:     {fix_count}")
    print(f"  ❌ CRITICAL_FAIL:  {fail_count}")

    # Coverage stats
    total_missing = 0
    for r in results:
        total_missing += len(r.get("coverage", {}).get("missing_from_modules", []))

    print(f"\nTotal missing items across all chapters: {total_missing}")

    # Accuracy stats
    wrong_formulas = sum(len(r.get("accuracy", {}).get("wrong_formulas", [])) for r in results)
    wrong_answers = sum(len(r.get("accuracy", {}).get("wrong_answers", [])) for r in results)
    hallucinated = sum(len(r.get("accuracy", {}).get("hallucinated_content", [])) for r in results)
    incomplete = sum(len(r.get("accuracy", {}).get("incomplete_examples", [])) for r in results)

    print(f"\nAccuracy issues:")
    print(f"  Wrong formulas:      {wrong_formulas}")
    print(f"  Wrong answers:       {wrong_answers}")
    print(f"  Hallucinated:        {hallucinated}")
    print(f"  Incomplete examples: {incomplete}")

    # List failures
    if fail_count > 0:
        print(f"\n❌ Critical failures:")
        for r in results:
            if r.get("overall_verdict") == "CRITICAL_FAIL":
                print(f"  {r.get('merge_code')}: {r.get('summary', '')}")

    if fix_count > 0:
        print(f"\n⚠ Needs fix:")
        for r in results:
            if r.get("overall_verdict") == "NEEDS_FIX":
                code = r.get("merge_code", "?")
                missing = r.get("coverage", {}).get("missing_from_modules", [])
                print(f"  {code}: {r.get('summary', '')} ({len(missing)} missing)")

    # Total modules
    total_modules = sum(r.get("total_modules", 0) for r in results)
    print(f"\nTotal modules across all chapters: {total_modules}")

    # Save summary
    summary = {
        "total_chapters": total,
        "pass": pass_count,
        "needs_fix": fix_count,
        "critical_fail": fail_count,
        "total_modules": total_modules,
        "total_missing_items": total_missing,
        "accuracy_issues": {
            "wrong_formulas": wrong_formulas,
            "wrong_answers": wrong_answers,
            "hallucinated": hallucinated,
            "incomplete_examples": incomplete
        },
        "failed_chapters": [
            {"merge_code": r.get("merge_code"), "summary": r.get("summary")}
            for r in results if r.get("overall_verdict") == "CRITICAL_FAIL"
        ],
        "needs_fix_chapters": [
            {"merge_code": r.get("merge_code"), "summary": r.get("summary"),
             "missing": r.get("coverage", {}).get("missing_from_modules", [])}
            for r in results if r.get("overall_verdict") == "NEEDS_FIX"
        ]
    }

    summary_path = os.path.join(OUTPUT_FOLDER, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary saved: {summary_path}")

# -----------------------------
# ASYNC RUNNER
# -----------------------------
async def main():

    merge_codes = sorted([
        d for d in os.listdir(MODULES_FOLDER)
        if os.path.isdir(os.path.join(MODULES_FOLDER, d))
    ])

    if DEBUG:
        merge_codes = merge_codes[:1]

    print(f"{'='*60}")
    print(f"🔍 MODULE VALIDATION (modules vs PDF)")
    print(f"{'='*60}")
    print(f"📦 {len(merge_codes)} chapters to validate\n")

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, validate_chapter, code)
            for code in merge_codes
        ]
        await asyncio.gather(*tasks)

    generate_summary()
    print("\n✅ Done")

if __name__ == "__main__":
    asyncio.run(main())
