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
EXTRACTED_FOLDER = "claude_works/extracted_text"
FIXED_FOLDER = "claude_works/extracted_text_fixed"
DATA_PATH = "data.json"
OUTPUT_FOLDER = "claude_works/validation_results"

MAX_WORKERS = 3
DEBUG = False       # True → only one file
FIX_MODE = True     # True → also fix FAIL/NEEDS_REVIEW files

# THRESHOLDS — only fix if scores drop below these
# Anything above = good enough, don't waste API calls
FIX_THRESHOLD = 80          # Only fix if ANY score < 80
REVALIDATE_PASS_THRESHOLD = 70  # After fix, PASS if all scores >= 70 (lenient)

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FIXED_FOLDER, exist_ok=True)

# -----------------------------
# LOAD DATA MAP
# -----------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Build merge_code → chapter info lookup
code_info = {}
for domain, chapters in data.items():
    for ch in chapters:
        code = ch["merge_code"]
        if code not in code_info:
            code_info[code] = {
                "domain": domain,
                "chapters": []
            }
        code_info[code]["chapters"].append({
            "class": ch["class"],
            "chapter_no": ch["chapter_no"],
            "chapter_name": ch["chapter_name"]
        })

# -----------------------------
# EXTRACTION RULES (for fix pass)
# -----------------------------
EXTRACTION_RULES = """
EXTRACTION FORMAT RULES (the fixed text MUST follow these exactly):

1. Preserve the original Kannada text as-is — do NOT translate or paraphrase
2. Mark headings clearly:
   # Chapter Title
   ## Section Heading
   ### Subsection Heading

3. Write all math formulas in LaTeX:
   Inline: $formula$
   Display: $$formula$$

4. Mark worked examples clearly:
   --- EXAMPLE START ---
   Source: Class {X}, Example {N} (page approximate)
   Problem: ...
   Solution:
   Step 1: ...
   Step 2: ...
   Answer: ...
   --- EXAMPLE END ---

5. Mark theorems/properties clearly:
   --- THEOREM ---
   Name: ...
   Statement: ...
   Proof: ... (if given, reproduce FULLY — do not summarize)
   --- END THEOREM ---

6. Mark definitions:
   --- DEFINITION ---
   Term: ...
   Definition: ...
   --- END DEFINITION ---

7. Tables as markdown:
   | Header 1 | Header 2 |
   |----------|----------|
   | data     | data     |

8. Figures — describe what you SEE:
   --- FIGURE ---
   Description: [describe the geometric figure, graph, or diagram exactly]
   Labels: [list ALL labeled points, angles, measurements, axes]
   Mathematical significance: [what concept this figure illustrates]
   --- END FIGURE ---

9. Important boxes / remarks:
    --- NOTE ---
    (content)
    --- END NOTE ---

WHAT TO EXTRACT:
✔ Chapter title and ALL section/subsection headings
✔ All body text — definitions, explanations, remarks, notes
✔ All formulas and equations — write in LaTeX
✔ All theorem statements and proofs (full, not summarized)
✔ All worked examples with COMPLETE solutions (every step)
✔ All solved examples within exercise sections
✔ All tables — reproduce as markdown tables
✔ All figure/diagram descriptions — describe what the figure shows
✔ All "Important" / "ನೆನಪಿಡಿ" / "ಗಮನಿಸಿ" boxes
✔ Properties, rules, and listed items
✔ Any derivation of formulas shown in the text

WHAT TO SKIP:
✗ UNSOLVED exercise questions (after "ಅಭ್ಯಾಸ" / "Exercise") — but KEEP solved examples within exercises
✗ Page numbers, headers, footers
✗ Publisher info, copyright text, preface
✗ Decorative images with no mathematical content
"""

# -----------------------------
# VALIDATION PROMPT
# -----------------------------
VALIDATE_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 VALIDATION — EXTRACTED TEXT vs SOURCE PDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE: You are a quality auditor. You have TWO inputs:
1. The ORIGINAL PDF (source of truth)
2. The EXTRACTED TEXT (output to validate)

Your job: compare them and find REAL problems — things that will break downstream
content generation. Do NOT nitpick minor formatting preferences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT MATTERS (report these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL — must fix:
- Entire sections/examples/theorems/definitions MISSING from extracted text
- Mathematical formulas that are WRONG (wrong numbers, wrong operators, wrong variables)
- Example solutions with MISSING steps or wrong answers
- Proofs that are truncated or summarized instead of full
- Content that is HALLUCINATED (not in the PDF at all)

MAJOR — should fix:
- Formulas with broken LaTeX that won't render (unbalanced $, broken commands)
- Marker tags not properly opened/closed (--- DEFINITION --- without --- END DEFINITION ---)
- Figures in PDF that have NO description at all in extracted text

IGNORE — do NOT report these:
- Minor wording differences that preserve meaning
- Slightly different figure description wording (as long as key info is captured)
- Extra whitespace, blank lines, minor formatting
- Heading level being ## vs ### (as long as hierarchy is roughly right)
- Having extra notes/remarks that don't hurt
- Minor LaTeX style differences ($x^2$ vs $x^{2}$) that render the same
- Section ordering being slightly different from PDF if content is all present

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — Return ONLY valid JSON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```json
{
  "merge_code": "<MERGE_CODE>",
  "overall_verdict": "PASS" | "FAIL" | "NEEDS_REVIEW",
  "confidence": 0.0-1.0,
  "summary": "One-line summary of overall quality",

  "scores": {
    "completeness": 0-100,
    "accuracy": 0-100,
    "formatting": 0-100,
    "exclusions": 0-100
  },

  "stats": {
    "sections_in_pdf": <int>,
    "sections_in_extracted": <int>,
    "definitions_in_pdf": <int>,
    "definitions_in_extracted": <int>,
    "theorems_in_pdf": <int>,
    "theorems_in_extracted": <int>,
    "examples_in_pdf": <int>,
    "examples_in_extracted": <int>,
    "figures_in_pdf": <int>,
    "figures_in_extracted": <int>,
    "formulas_in_pdf_approx": <int>,
    "formulas_in_extracted_approx": <int>
  },

  "issues": [
    {
      "check": "completeness" | "accuracy" | "formatting" | "exclusions",
      "severity": "critical" | "major",
      "description": "What is wrong",
      "location": "Where in the text/PDF",
      "expected": "What should be there (if applicable)",
      "actual": "What is there instead (if applicable)"
    }
  ],

  "missing_content": [
    "List of specific items missing from extracted text"
  ],

  "unwanted_content": [
    "List of items that should NOT be in extracted text but are"
  ]
}
```

SCORING RULES:
- "PASS" = all scores >= 90 AND no critical issues
- "FAIL" = any score < 70 OR 2+ critical issues
- "NEEDS_REVIEW" = everything else
- Only report "critical" and "major" severity. Skip minor issues entirely.
- Return ONLY the JSON. No commentary before or after.
"""

# -----------------------------
# FIX PROMPT
# -----------------------------
FIX_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 FIX — CORRECT THE EXTRACTED TEXT USING THE SOURCE PDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE: You are a precise text corrector. You have THREE inputs:
1. The ORIGINAL PDF (source of truth)
2. The EXTRACTED TEXT (current version with issues)
3. The VALIDATION REPORT (lists the critical/major issues found)

Your job: produce a CORRECTED version of the extracted text.
Focus ONLY on fixing the critical and major issues listed in the validation report.
Do NOT rewrite parts that are already correct.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO FIX (only these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MISSING CONTENT — Add any sections, definitions, theorems, examples, figures
   that are listed as missing. Extract them from the PDF.

2. WRONG FORMULAS — Correct any LaTeX that doesn't match the PDF.

3. INCOMPLETE EXAMPLES — Fill in missing solution steps from the PDF.

4. INCOMPLETE PROOFS — Add missing proof steps from the PDF.

5. BROKEN FORMATTING — Fix unclosed marker tags.

6. HALLUCINATED CONTENT — Remove anything not in the PDF.

DO NOT touch anything that's already correct. Keep the structure stable.

{extraction_rules}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* The PDF is the ONLY source of truth — do NOT invent content
* Keep everything that was CORRECT — only fix what's listed in the report
* Do NOT rephrase correct Kannada text
* Do NOT reorder sections unless the report says ordering is wrong
* Mathematical accuracy is CRITICAL — double-check every formula against the PDF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output the COMPLETE corrected extracted text.
No JSON wrapper. No commentary. Just the full corrected text from start to finish.
"""

# ═══════════════════════════════════════
# REVALIDATION PROMPT (lenient, post-fix)
# ═══════════════════════════════════════
REVALIDATE_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST-FIX VALIDATION — Quick check of corrected text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLE: You are doing a QUICK final check of corrected extracted text against the PDF.
This is the LAST pass — be practical, not perfectionist.

Only flag issues that are genuinely broken:
- Entire sections/examples still MISSING
- Formulas that are mathematically WRONG (not style differences)
- Hallucinated content not in the PDF
- Completely broken marker formatting

Do NOT flag:
- Minor wording differences
- Style preferences
- Small formatting inconsistencies
- Anything cosmetic

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — Return ONLY valid JSON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```json
{
  "merge_code": "<MERGE_CODE>",
  "overall_verdict": "PASS" | "ACCEPTABLE" | "STILL_BROKEN",
  "confidence": 0.0-1.0,
  "summary": "One-line summary",

  "scores": {
    "completeness": 0-100,
    "accuracy": 0-100,
    "formatting": 0-100,
    "exclusions": 0-100
  },

  "remaining_issues": [
    {
      "severity": "critical",
      "description": "What is still wrong",
      "location": "Where"
    }
  ]
}
```

RULES:
- "PASS" = all good, no real issues left
- "ACCEPTABLE" = minor things remain but usable for downstream processing
- "STILL_BROKEN" = critical content still missing or wrong — needs manual review
- Only report critical issues. If you find yourself listing minor things, just say PASS.
- Return ONLY the JSON.
"""

# -----------------------------
# HELPERS
# -----------------------------
def needs_fixing(validation):
    """Check if a file actually needs fixing based on scores."""
    scores = validation.get("scores", {})
    # Only fix if any score is below threshold
    for key in ["completeness", "accuracy", "formatting", "exclusions"]:
        if scores.get(key, 100) < FIX_THRESHOLD:
            return True
    # Or if there are critical issues
    for issue in validation.get("issues", []):
        if issue.get("severity") == "critical":
            return True
    return False

def parse_json_response(text):
    """Parse JSON from LLM response, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)

def get_chapters_str(merge_code):
    info = code_info.get(merge_code, {})
    if not info:
        return ""
    chapters_str = f"Domain: {info['domain']}\n"
    for ch in info["chapters"]:
        chapters_str += f"  - Class {ch['class']}, Ch {ch['chapter_no']}: {ch['chapter_name']}\n"
    return chapters_str

def read_pdf_part(pdf_path):
    with open(pdf_path, "rb") as f:
        return Part.from_data(data=f.read(), mime_type="application/pdf")

# -----------------------------
# VALIDATE SINGLE FILE
# -----------------------------
def validate_file(merge_code):

    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    txt_path = os.path.join(EXTRACTED_FOLDER, f"{merge_code}.txt")
    output_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_validation.json")

    if os.path.exists(output_path):
        print(f"⏭ Skipping validation: {merge_code}")
        return

    if not os.path.exists(pdf_path):
        print(f"⚠ No PDF for: {merge_code}")
        return

    if not os.path.exists(txt_path):
        print(f"⚠ No extracted text for: {merge_code}")
        return

    print(f"🔍 Validating: {merge_code}")

    pdf_part = read_pdf_part(pdf_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        extracted_text = f.read()

    prompt = f"""MERGE_CODE: {merge_code}
{get_chapters_str(merge_code)}

--- EXTRACTED TEXT (to validate) ---
{extracted_text}
--- END EXTRACTED TEXT ---

{VALIDATE_PROMPT}"""

    try:
        response = model.generate_content(
            [pdf_part, prompt],
            generation_config={"temperature": 0.1}
        )

        result = parse_json_response(response.text)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        verdict = result.get("overall_verdict", "?")
        scores = result.get("scores", {})
        issues_count = len(result.get("issues", []))

        print(f"  ✅ {merge_code}: {verdict} | "
              f"C:{scores.get('completeness','?')} "
              f"A:{scores.get('accuracy','?')} "
              f"F:{scores.get('formatting','?')} "
              f"E:{scores.get('exclusions','?')} | "
              f"{issues_count} issues")

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
# FIX SINGLE FILE
# -----------------------------
def fix_file(merge_code):

    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    txt_path = os.path.join(EXTRACTED_FOLDER, f"{merge_code}.txt")
    validation_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_validation.json")
    fixed_path = os.path.join(FIXED_FOLDER, f"{merge_code}.txt")
    revalidation_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_revalidation.json")

    # Skip if already done (fixed + revalidated)
    if os.path.exists(revalidation_path):
        print(f"⏭ Skipping (already done): {merge_code}")
        return
    if os.path.exists(fixed_path):
        print(f"⏭ Skipping fix: {merge_code}")
        # Still need revalidation? Check below.
        if not os.path.exists(revalidation_path):
            revalidate_file(merge_code)
        return

    # Load validation report
    if not os.path.exists(validation_path):
        print(f"⚠ No validation report for: {merge_code}")
        return

    with open(validation_path, "r", encoding="utf-8") as f:
        validation = json.load(f)

    # Check if it actually needs fixing
    if not needs_fixing(validation):
        # Good enough — copy as-is
        with open(txt_path, "r", encoding="utf-8") as f:
            original = f.read()
        with open(fixed_path, "w", encoding="utf-8") as f:
            f.write(original)
        # Save a simple revalidation = same as original validation
        with open(revalidation_path, "w", encoding="utf-8") as f:
            json.dump({
                "merge_code": merge_code,
                "overall_verdict": "PASS",
                "summary": "Scores above threshold — no fix needed",
                "scores": validation.get("scores", {}),
                "remaining_issues": [],
                "skipped_fix": True
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ {merge_code}: Good enough (scores >= {FIX_THRESHOLD}) — copied as-is")
        return

    if not os.path.exists(pdf_path):
        print(f"⚠ No PDF for fix: {merge_code}")
        return

    verdict = validation.get("overall_verdict", "")
    print(f"🔧 Fixing: {merge_code} ({verdict})")

    pdf_part = read_pdf_part(pdf_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        extracted_text = f.read()

    # Only include critical + major issues in fix prompt (skip minor)
    real_issues = [i for i in validation.get("issues", []) if i.get("severity") in ("critical", "major")]
    issues_text = json.dumps(real_issues, indent=2, ensure_ascii=False)
    missing_text = json.dumps(validation.get("missing_content", []), indent=2, ensure_ascii=False)
    unwanted_text = json.dumps(validation.get("unwanted_content", []), indent=2, ensure_ascii=False)

    fix_prompt_formatted = FIX_PROMPT.replace("{extraction_rules}", EXTRACTION_RULES)

    prompt = f"""MERGE_CODE: {merge_code}

--- ORIGINAL EXTRACTED TEXT (has issues) ---
{extracted_text}
--- END EXTRACTED TEXT ---

--- VALIDATION REPORT ---
Verdict: {verdict}
Summary: {validation.get('summary', '')}

Scores:
  Completeness: {validation.get('scores', {}).get('completeness', '?')}
  Accuracy: {validation.get('scores', {}).get('accuracy', '?')}
  Formatting: {validation.get('scores', {}).get('formatting', '?')}
  Exclusions: {validation.get('scores', {}).get('exclusions', '?')}

Critical/Major Issues ({len(real_issues)} total):
{issues_text}

Missing Content:
{missing_text}

Unwanted Content:
{unwanted_text}
--- END VALIDATION REPORT ---

{fix_prompt_formatted}"""

    try:
        response = model.generate_content(
            [pdf_part, prompt],
            generation_config={"temperature": 0.1}
        )

        fixed_text = response.text.strip()

        with open(fixed_path, "w", encoding="utf-8") as f:
            f.write(fixed_text)

        original_lines = len(extracted_text.splitlines())
        fixed_lines = len(fixed_text.splitlines())
        diff = fixed_lines - original_lines

        print(f"  🔧 {merge_code}: Fixed ({original_lines} → {fixed_lines} lines, "
              f"{'+' if diff >= 0 else ''}{diff})")

        # Now revalidate the fixed text
        revalidate_file(merge_code)

    except Exception as e:
        error_path = os.path.join(FIXED_FOLDER, f"{merge_code}_FIX_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"  ❌ {merge_code} fix failed: {e}")

# -----------------------------
# REVALIDATE FIXED FILE
# -----------------------------
def revalidate_file(merge_code):

    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    fixed_path = os.path.join(FIXED_FOLDER, f"{merge_code}.txt")
    revalidation_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_revalidation.json")

    if os.path.exists(revalidation_path):
        print(f"  ⏭ Skipping revalidation: {merge_code}")
        return

    if not os.path.exists(fixed_path) or not os.path.exists(pdf_path):
        return

    print(f"  🔄 Revalidating: {merge_code}")

    pdf_part = read_pdf_part(pdf_path)

    with open(fixed_path, "r", encoding="utf-8") as f:
        fixed_text = f.read()

    prompt = f"""MERGE_CODE: {merge_code}
{get_chapters_str(merge_code)}

--- CORRECTED EXTRACTED TEXT (to check) ---
{fixed_text}
--- END EXTRACTED TEXT ---

{REVALIDATE_PROMPT}"""

    try:
        response = model.generate_content(
            [pdf_part, prompt],
            generation_config={"temperature": 0.1}
        )

        result = parse_json_response(response.text)

        with open(revalidation_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        verdict = result.get("overall_verdict", "?")
        scores = result.get("scores", {})
        remaining = len(result.get("remaining_issues", []))

        print(f"  🔄 {merge_code}: {verdict} | "
              f"C:{scores.get('completeness','?')} "
              f"A:{scores.get('accuracy','?')} "
              f"F:{scores.get('formatting','?')} "
              f"E:{scores.get('exclusions','?')} | "
              f"{remaining} remaining issues")

    except json.JSONDecodeError:
        error_path = os.path.join(OUTPUT_FOLDER, f"{merge_code}_revalidation_RAW.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  ⚠ {merge_code}: Revalidation JSON parse error")

    except Exception as e:
        print(f"  ⚠ {merge_code}: Revalidation failed — {e}")

# -----------------------------
# GENERATE SUMMARY
# -----------------------------
def generate_summary():
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)

    # Initial validation results
    initial_results = []
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith("_validation.json"):
            with open(os.path.join(OUTPUT_FOLDER, f), "r", encoding="utf-8") as fh:
                initial_results.append(json.load(fh))

    if not initial_results:
        print("No validation results found.")
        return

    total = len(initial_results)

    # Initial stats
    init_pass = sum(1 for r in initial_results if r.get("overall_verdict") == "PASS")
    init_fail = sum(1 for r in initial_results if r.get("overall_verdict") == "FAIL")
    init_review = sum(1 for r in initial_results if r.get("overall_verdict") == "NEEDS_REVIEW")

    print(f"\n--- PASS 1: Initial Validation ---")
    print(f"Total: {total}")
    print(f"  PASS:         {init_pass}")
    print(f"  FAIL:         {init_fail}")
    print(f"  NEEDS_REVIEW: {init_review}")

    # Revalidation results
    reval_results = []
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith("_revalidation.json"):
            with open(os.path.join(OUTPUT_FOLDER, f), "r", encoding="utf-8") as fh:
                reval_results.append(json.load(fh))

    if reval_results and FIX_MODE:
        skipped = sum(1 for r in reval_results if r.get("skipped_fix"))
        fixed = len(reval_results) - skipped

        reval_pass = sum(1 for r in reval_results if r.get("overall_verdict") in ("PASS", "ACCEPTABLE"))
        reval_broken = sum(1 for r in reval_results if r.get("overall_verdict") == "STILL_BROKEN")

        print(f"\n--- PASS 2: Fix + Revalidation ---")
        print(f"  Skipped (good enough): {skipped}")
        print(f"  Fixed:                 {fixed}")
        print(f"  After fix — PASS/OK:   {reval_pass}")
        print(f"  After fix — BROKEN:    {reval_broken}")

        # List still-broken files
        if reval_broken > 0:
            print(f"\n⚠ Still broken (need manual review):")
            for r in reval_results:
                if r.get("overall_verdict") == "STILL_BROKEN":
                    code = r.get("merge_code", "?")
                    summary = r.get("summary", "")
                    print(f"  {code}: {summary}")

    # Final output location
    fixed_count = len([
        f for f in os.listdir(FIXED_FOLDER)
        if f.endswith(".txt") and not f.endswith("_FIX_ERROR.txt")
    ]) if os.path.exists(FIXED_FOLDER) else 0

    print(f"\n📁 Final output: {FIXED_FOLDER}/ ({fixed_count} files)")

    # Save summary JSON
    summary_data = {
        "total": total,
        "initial": {"pass": init_pass, "fail": init_fail, "needs_review": init_review},
        "after_fix": {
            "total_revalidated": len(reval_results),
            "pass_or_acceptable": sum(1 for r in reval_results if r.get("overall_verdict") in ("PASS", "ACCEPTABLE")),
            "still_broken": sum(1 for r in reval_results if r.get("overall_verdict") == "STILL_BROKEN"),
            "still_broken_files": [
                {"merge_code": r.get("merge_code"), "summary": r.get("summary")}
                for r in reval_results if r.get("overall_verdict") == "STILL_BROKEN"
            ]
        },
        "fixed_files_count": fixed_count
    }

    summary_path = os.path.join(OUTPUT_FOLDER, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary saved: {summary_path}")

# -----------------------------
# ASYNC RUNNER
# -----------------------------
async def main():

    txt_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(EXTRACTED_FOLDER)
        if f.endswith(".txt") and not f.endswith("_ERROR.txt")
    ]
    txt_files.sort()

    if DEBUG:
        txt_files = txt_files[:1]

    # ── PASS 1: VALIDATE ──
    print(f"{'='*60}")
    print(f"📋 PASS 1: VALIDATION")
    print(f"{'='*60}")
    print(f"📦 Found {len(txt_files)} extracted text files\n")

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, validate_file, code)
            for code in txt_files
        ]
        await asyncio.gather(*tasks)

    # ── PASS 2: FIX + REVALIDATE (one shot, no loop) ──
    if FIX_MODE:
        print(f"\n{'='*60}")
        print(f"🔧 PASS 2: FIX + REVALIDATE")
        print(f"{'='*60}")
        print(f"Threshold: only fix if any score < {FIX_THRESHOLD}\n")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [
                loop.run_in_executor(executor, fix_file, code)
                for code in txt_files
            ]
            await asyncio.gather(*tasks)

    generate_summary()

    print("\n✅ All done — no loops, no retries.")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
