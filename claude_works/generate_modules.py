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

EXTRACTED_FOLDER = "claude_works/extracted_text_fixed"
PDF_FOLDER = "merged"
DATA_PATH = "data.json"
MODULES_FOLDER = "claude_works/modules"
MASTER_DATA_FOLDER = "claude_works/master_data"

MAX_WORKERS = 3
DEBUG = False       # True → only one file

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(MODULES_FOLDER, exist_ok=True)
os.makedirs(MASTER_DATA_FOLDER, exist_ok=True)

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
# MASTER DATA PROMPT
# -----------------------------
MASTER_DATA_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTED TEXT → STRUCTURED MASTER DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ನೀವು ಕನ್ನಡ ಮಾಧ್ಯಮ 6–10 ಗಣಿತ ಪಠ್ಯಪುಸ್ತಕ ಆಧಾರಿತ structured content generator ಆಗಿದ್ದೀರಿ.

INPUT:  ಒಂದು topic ನ clean extracted text (one or more chapters merged by topic)
        + ORIGINAL PDF for visual verification
OUTPUT: GPSTR/HSTR ಪರೀಕ್ಷೆಗೆ ಅನುಗುಣವಾದ structured master data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET EXAM: GPSTR / HSTR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPSTR (Graduate Primary School Teacher Recruitment) ಮತ್ತು
HSTR (High School Teacher Recruitment) — ಕರ್ನಾಟಕ ರಾಜ್ಯ ಶಿಕ್ಷಕರ ನೇಮಕಾತಿ ಪರೀಕ್ಷೆ.

EXAM PATTERN:
* MCQ ಮತ್ತು short/long answer ಪ್ರಶ್ನೆಗಳು
* Content coverage: Class 6–10 NCERT/State board syllabus
* Medium: ಕನ್ನಡ
* Focus: conceptual understanding + problem solving + theorem application

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOPIC-WISE CONTENT:
Input may contain content from multiple classes merged together. Handle by:
* ಸರಳದಿಂದ ಕಠಿಣಕ್ಕೆ ಕ್ರಮ (foundational → advanced)
* ಒಂದೇ ಸೂತ್ರ / ಕಲ್ಪನೆಯ ಪುನರಾವರ್ತನೆ ತಪ್ಪಿಸಬೇಕು
* Repeated theorems, formulas → ONE consolidated explanation

CONTENT FILTERING:
IGNORE:
✗ ಪಾಠದ ಪರಿಚಯ
✗ Learning Outcomes
✗ Activity sections (UNLESS the activity derives a formula)
✗ Trivial "ತಿಳಿದಿರಲಿ / Did You Know" boxes

EXTRACT:
✔ ALL Definitions — with exact Kannada + English term
✔ ALL Formulas — with LaTeX, derivation if shown in text, conditions, exceptions
✔ ALL Theorems — with FULL proof if text has it, converse, special cases
✔ ALL Properties and Rules
✔ ALL Worked Examples — COMPLETE step-by-step, EVERY step, EVERY calculation
✔ ALL Special Cases, Exceptions, Boundary Conditions
✔ ALL Figures/Diagrams — preserve descriptions, labels, mathematical significance
✔ ALL Tables — preserve as markdown tables
✔ Common mistakes students make

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMULA RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

formula_name: ""
latex: ""                    ← character-perfect from textbook
condition: ""                ← when can this formula be used?
exception: ""                ← when does it NOT work?
derived_from: ""             ← which simpler formula leads to this?
used_in: ""                  ← what kind of problems use this?

If textbook shows derivation, include full steps:
derivation:
  - step: 1
    action: ""
    calculation: ""
  result: ""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEOREM RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

theorem_name: ""
statement: ""                ← faithful to textbook
proof: ""                    ← FULL proof if textbook has it, else omit
converse: ""                 ← if mentioned
special_cases: []
common_error: ""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

example_id: "{MERGE_CODE}_example_{N}"
difficulty: "basic" | "intermediate" | "advanced"
problem: ""
steps:
  - step: 1
    action: ""              ← what you're doing in this step
    justification: ""       ← why this step
    calculation: ""         ← the actual math computation
result: ""
common_trap: ""

DIFFICULTY GUIDE:
* basic       → direct formula substitution, 1–2 steps
* intermediate → requires choosing the right formula, 3–4 steps
* advanced    → multi-concept, 5+ steps, boundary checks, exam-style

CRITICAL:
* Show ALL steps. Never write "ಅದೇ ರೀತಿ..." or "similarly..."
* Include EVERY worked example from the extracted text — do NOT skip any
* Every step MUST have a calculation field showing the actual math

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL AIDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Include ALL figures and tables from the extracted text. Check the PDF for any
the text extraction may have missed.

[ದೃಶ್ಯ_ಸಹಾಯ]
type: diagram | table | graph | number_line
description: (what the figure shows — detailed enough to recreate)
labels: (ALL points, measurements, angles, axes)
mathematical_significance: (what concept this illustrates)

For tables, also include the full markdown table data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT SEGMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* Each major concept = separate [ಮೂಲ_ಅವಧಾರಣೆ] block
* Too big (5+ definitions OR 3+ theorems) → split into sub-concepts
* Too small (1 definition, no examples) → merge with related concept
* Order: foundational → advanced (teaching flow)
* No limit on number of concepts — cover EVERYTHING in the source text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* ಎಲ್ಲಾ ವಿವರಣೆಗಳು ಕನ್ನಡದಲ್ಲೇ ಇರಬೇಕು
* ಗಣಿತ terms: ಕನ್ನಡ ಪದ (English term) — ಉದಾ: "ಸಮಾಂತರ ಶ್ರೇಢಿ (Arithmetic Progression)"
* LaTeX, formula names → English OK
* ಗಣಿತ ನಿಖರತೆ ಕಡ್ಡಾಯ — ಯಾವುದೇ ಸೂತ್ರ ತಪ್ಪಾಗಿ ದಾಖಲಿಸಬಾರದು
* ವ್ಯಾಖ್ಯೆಗಳು faithful to textbook
* Do NOT hallucinate — every piece of content must come from the source text or PDF
* Cross-check formulas and example answers against the PDF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: Use this EXACT format. Do NOT output JSON.

[ಅಧ್ಯಾಯ_ಮೆಟಾಡೇಟಾ]
merge_code:
subject: ಗಣಿತ
topic:
domain:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ಮೂಲ_ಅವಧಾರಣೆ_1]
concept_id: {MERGE_CODE}_concept_1
concept_name: ಕನ್ನಡ ಹೆಸರು (English name)
domain:

  [ಸಾರಾಂಶ]
  (2–4 sentences — teacher-style overview)

  [ವ್ಯಾಖ್ಯೆ]
  (all definitions for this concept)

  [ಸೂತ್ರಗಳು]
  (all formulas with LaTeX, conditions, derivations)

  [ಪ್ರಮೇಯ]
  (all theorems with proofs)

  [ಅಂಶಪಟ್ಟಿ / Properties]
  (all properties and rules)

  [ಕ್ರಮಬದ್ಧ ಉದಾಹರಣೆ]
  (ALL worked examples with COMPLETE steps + calculations)

  [ದೃಶ್ಯ_ಸಹಾಯ]
  (figures, tables, graphs — with full descriptions)

  [ವಿಶ್ಲೇಷಣೆ]
  — ಈ ಅವಧಾರಣೆ ಏಕೆ ಮುಖ್ಯ
  — Boundary conditions
  — Common mistakes

  [ಸಂಬಂಧ]
  — ಪೂರ್ವಾಪೇಕ್ಷಿತ ಅವಧಾರಣೆಗಳು
  — ಮುಂದಿನ ಅವಧಾರಣೆಗಳೊಂದಿಗೆ ಸಂಪರ್ಕ

  [ಶೈಕ್ಷಣಿಕ_ದೃಷ್ಟಿಕೋನ]
  — gpstr_weightage: high | medium | low
  — MCQ ಸಾಧ್ಯತೆ:
  — 2-ಅಂಕ ಪ್ರಶ್ನೆ:
  — 5-ಅಂಕ ಪ್ರಶ್ನೆ:
  — ಪ್ರಮುಖ trap:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ಮೂಲ_ಅವಧಾರಣೆ_2]
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Content_Coverage_Check]
* ಒಟ್ಟು ಅವಧಾರಣೆಗಳು:
* ಒಟ್ಟು definitions:
* ಒಟ್ಟು formulas:
* ಒಟ್ಟು theorems:
* ಒಟ್ಟು worked examples:
* ಒಟ್ಟು figures/tables:
* ಎಲ್ಲಾ ಉಪಶೀರ್ಷಿಕೆಗಳು cover ಆಗಿವೆಯೆ?: Yes/No
"""

# -----------------------------
# MODULE SPLIT PROMPT
# -----------------------------
MODULE_SPLIT_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MASTER DATA → MODULE JSON FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ನೀವು ಕನ್ನಡ ಮಾಧ್ಯಮ GPSTR ಗಣಿತ LMS module splitter ಆಗಿದ್ದೀರಿ.

INPUT:  ಒಂದು chapter ನ master data (structured text with [ಮೂಲ_ಅವಧಾರಣೆ] blocks)
OUTPUT: JSON object — ಪ್ರತಿ [ಮೂಲ_ಅವಧಾರಣೆ] → ಒಂದು standalone module

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPLITTING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* ಪ್ರತಿ [ಮೂಲ_ಅವಧಾರಣೆ_{N}] → ಒಂದು separate JSON file
* concept_id master data ದಿಂದ ನೇರ copy
* ಒಂದು concept ನ content ಬೇರೆ module ಗೆ leak ಆಗಬಾರದು
* ಕ್ರಮ master data ಅನುಸರಿಸಬೇಕು
* prerequisites → ಅದೇ chapter ನ ಹಿಂದಿನ concept_id ಗಳ list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE JSON SCHEMA (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each module must follow this exact schema:

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
    "theorems": [
      {
        "theorem_name": "",
        "statement": "",
        "proof": "",
        "converse": "",
        "special_cases": [],
        "common_error": ""
      }
    ],
    "properties": [],
    "visual_aids": [
      {
        "type": "diagram | table | graph | number_line",
        "description": "",
        "labels": "",
        "mathematical_significance": "",
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

EXTRACT FAITHFULLY:
✔ [ವ್ಯಾಖ್ಯೆ]           → theory.definitions + theory.concept_summary
✔ [ಸೂತ್ರಗಳು]          → theory.formulas (LaTeX ನೇರ copy — do NOT change)
✔ [ಪ್ರಮೇಯ]            → theory.theorems (with FULL proof)
✔ [ಅಂಶಪಟ್ಟಿ]          → theory.properties
✔ [ಕ್ರಮಬದ್ಧ ಉದಾಹರಣೆ]  → worked_examples (ALL steps + ALL calculations)
✔ [ವಿಶ್ಲೇಷಣೆ]         → exam_intelligence (mistakes, boundary conditions)
✔ [ಶೈಕ್ಷಣಿಕ_ದೃಷ್ಟಿಕೋನ] → exam_intelligence (weightage, mcq/2mark/5mark)
✔ [ದೃಶ್ಯ_ಸಹಾಯ]        → theory.visual_aids

CRITICAL:
* LaTeX formulas — character-perfect copy from master data
* Every worked example step MUST have calculation field
* Do NOT hallucinate — zero invented content
* concept_summary must be a paraphrase, NOT a direct copy of definitions
* visual_aids must include ALL figures, tables from master data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output a JSON object with concept_ids as keys:

{
  "{MERGE_CODE}_concept_1": { ...module JSON... },
  "{MERGE_CODE}_concept_2": { ...module JSON... },
  ...
}

* Valid JSON — no trailing commas, no comments
* Return ONLY the JSON. No commentary before or after.
"""

# -----------------------------
# HELPERS
# -----------------------------
def get_chapters_str(merge_code):
    info = code_info.get(merge_code, {})
    if not info:
        return ""
    result = f"Domain: {info['domain']}\n"
    for ch in info["chapters"]:
        result += f"  - Class {ch['class']}, Ch {ch['chapter_no']}: {ch['chapter_name']}\n"
    return result

def read_pdf_part(pdf_path):
    with open(pdf_path, "rb") as f:
        return Part.from_data(data=f.read(), mime_type="application/pdf")

def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)

# ═════════════════════════════════════════
# STEP 1: GENERATE MASTER DATA
# ═════════════════════════════════════════
def generate_master_data(merge_code):

    txt_path = os.path.join(EXTRACTED_FOLDER, f"{merge_code}.txt")
    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    output_path = os.path.join(MASTER_DATA_FOLDER, f"{merge_code}.txt")

    if os.path.exists(output_path):
        print(f"⏭ Skipping master data: {merge_code}")
        return True

    if not os.path.exists(txt_path):
        print(f"⚠ No extracted text for: {merge_code}")
        return False

    print(f"📝 Generating master data: {merge_code}")

    with open(txt_path, "r", encoding="utf-8") as f:
        extracted_text = f.read()

    parts = []

    # Include PDF for visual cross-check if available
    if os.path.exists(pdf_path):
        parts.append(read_pdf_part(pdf_path))

    prompt = f"""MERGE_CODE: {merge_code}
{get_chapters_str(merge_code)}

--- EXTRACTED TEXT ---
{extracted_text}
--- END EXTRACTED TEXT ---

{MASTER_DATA_PROMPT}"""

    parts.append(prompt)

    try:
        response = model.generate_content(
            parts,
            generation_config={"temperature": 0.1}
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        # Quick stats
        concepts = response.text.count("[ಮೂಲ_ಅವಧಾರಣೆ_")
        examples = response.text.count("example_id:")
        print(f"  ✅ {merge_code}: {concepts} concepts, {examples} examples")
        return True

    except Exception as e:
        error_path = os.path.join(MASTER_DATA_FOLDER, f"{merge_code}_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"  ❌ {merge_code}: {e}")
        return False

# ═════════════════════════════════════════
# STEP 2: SPLIT TO MODULES
# ═════════════════════════════════════════
def split_to_modules(merge_code):

    master_path = os.path.join(MASTER_DATA_FOLDER, f"{merge_code}.txt")
    module_dir = os.path.join(MODULES_FOLDER, merge_code)

    # Skip if module folder already has JSON files
    if os.path.exists(module_dir):
        existing = [f for f in os.listdir(module_dir) if f.endswith(".json")]
        if existing:
            print(f"⏭ Skipping modules: {merge_code} ({len(existing)} modules exist)")
            return

    if not os.path.exists(master_path):
        print(f"⚠ No master data for: {merge_code}")
        return

    print(f"📦 Splitting modules: {merge_code}")

    with open(master_path, "r", encoding="utf-8") as f:
        master_data = f.read()

    # Also pass the PDF for cross-checking formulas/figures
    pdf_path = os.path.join(PDF_FOLDER, f"{merge_code}.pdf")
    parts = []
    if os.path.exists(pdf_path):
        parts.append(read_pdf_part(pdf_path))

    prompt = f"""MERGE_CODE: {merge_code}
{get_chapters_str(merge_code)}

--- MASTER DATA ---
{master_data}
--- END MASTER DATA ---

{MODULE_SPLIT_PROMPT}"""

    parts.append(prompt)

    try:
        response = model.generate_content(
            parts,
            generation_config={"temperature": 0.1}
        )

        modules = parse_json_response(response.text)

        # Save each module as a separate file
        os.makedirs(module_dir, exist_ok=True)

        if isinstance(modules, dict):
            for concept_id, module_data in modules.items():
                module_path = os.path.join(module_dir, f"{concept_id}.json")
                with open(module_path, "w", encoding="utf-8") as f:
                    json.dump(module_data, f, indent=2, ensure_ascii=False)

            print(f"  ✅ {merge_code}: {len(modules)} modules saved")

        elif isinstance(modules, list):
            # Handle if model returns array instead of object
            for i, module_data in enumerate(modules):
                concept_id = module_data.get("module_id", f"{merge_code}_concept_{i+1}")
                module_path = os.path.join(module_dir, f"{concept_id}.json")
                with open(module_path, "w", encoding="utf-8") as f:
                    json.dump(module_data, f, indent=2, ensure_ascii=False)

            print(f"  ✅ {merge_code}: {len(modules)} modules saved")

    except json.JSONDecodeError:
        os.makedirs(module_dir, exist_ok=True)
        error_path = os.path.join(module_dir, f"{merge_code}_RAW.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  ⚠ {merge_code}: JSON parse error — raw saved")

    except Exception as e:
        os.makedirs(module_dir, exist_ok=True)
        error_path = os.path.join(module_dir, f"{merge_code}_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"  ❌ {merge_code}: {e}")

# ═════════════════════════════════════════
# FULL PIPELINE FOR ONE FILE
# ═════════════════════════════════════════
def process_file(merge_code):
    # Step 1: Generate master data
    success = generate_master_data(merge_code)
    if not success:
        return

    # Step 2: Split to modules
    split_to_modules(merge_code)

# ═════════════════════════════════════════
# GENERATE SUMMARY
# ═════════════════════════════════════════
def generate_summary():
    print("\n" + "=" * 60)
    print("📊 MODULE GENERATION SUMMARY")
    print("=" * 60)

    total_modules = 0
    chapter_stats = {}
    errors = []

    for merge_code in sorted(os.listdir(MODULES_FOLDER)):
        module_dir = os.path.join(MODULES_FOLDER, merge_code)
        if not os.path.isdir(module_dir):
            continue

        json_files = [f for f in os.listdir(module_dir) if f.endswith(".json")]
        error_files = [f for f in os.listdir(module_dir) if f.endswith("_ERROR.txt") or f.endswith("_RAW.txt")]

        chapter_stats[merge_code] = len(json_files)
        total_modules += len(json_files)

        if error_files:
            errors.append(merge_code)

    print(f"\nTotal chapters: {len(chapter_stats)}")
    print(f"Total modules:  {total_modules}")

    if errors:
        print(f"\n⚠ Errors in: {', '.join(errors)}")

    # Per-chapter breakdown
    print(f"\nPer chapter:")
    for code, count in sorted(chapter_stats.items()):
        status = "⚠" if code in errors else "✅"
        print(f"  {status} {code}: {count} modules")

    # Master data stats
    md_count = len([f for f in os.listdir(MASTER_DATA_FOLDER) if f.endswith(".txt") and not f.endswith("_ERROR.txt")])
    md_errors = len([f for f in os.listdir(MASTER_DATA_FOLDER) if f.endswith("_ERROR.txt")])

    print(f"\nMaster data: {md_count} generated, {md_errors} errors")

    # Save summary
    summary = {
        "total_chapters": len(chapter_stats),
        "total_modules": total_modules,
        "chapters": chapter_stats,
        "errors": errors,
        "master_data_count": md_count,
        "master_data_errors": md_errors
    }

    summary_path = os.path.join(MODULES_FOLDER, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary saved: {summary_path}")

# ═════════════════════════════════════════
# ASYNC RUNNER
# ═════════════════════════════════════════
async def main():

    txt_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(EXTRACTED_FOLDER)
        if f.endswith(".txt") and not f.endswith("_ERROR.txt")
    ]
    txt_files.sort()

    if DEBUG:
        txt_files = txt_files[:1]

    print(f"{'='*60}")
    print(f"📋 EXTRACTED TEXT → MASTER DATA → MODULES")
    print(f"{'='*60}")
    print(f"📦 Found {len(txt_files)} files to process\n")

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, process_file, code)
            for code in txt_files
        ]
        await asyncio.gather(*tasks)

    generate_summary()

    print("\n✅ All done")

# ═════════════════════════════════════════
# RUN
# ═════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())
