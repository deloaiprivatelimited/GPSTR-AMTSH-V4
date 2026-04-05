"""
Generate video lesson chunks (slides) from science module JSONs.
- Watches modules_science/ for new module folders (polling)
- Each module → intro + definitions + laws + experiments + formulas + examples + recap slides
- Uses ALL credentials (4 projects), 4 threads per project = 16 concurrent
- Structured JSON output via Gemini schema enforcement
"""
import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
LOCATION = "us-central1"
CREDENTIALS_DIR = Path("claude_works/credentials")
USE_ONLY = []  # empty = use ALL credentials

MODULES_FOLDER = Path("claude_works/modules_science")
CHUNKS_FOLDER = Path("claude_works/chunks_science")

THREADS_PER_PROJECT = 4
POLL_INTERVAL = 30
MAX_POLLS = 200
DEBUG = False

CHUNKS_FOLDER.mkdir(parents=True, exist_ok=True)

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
_thread_project_idx = threading.local()

def get_model(project_idx):
    key = f"model_{project_idx}"
    if not hasattr(_thread_local, key):
        p = PROJECTS[project_idx]
        vertexai.init(project=p["project_id"], location=LOCATION, credentials=p["credentials"])
        setattr(_thread_local, key, GenerativeModel("gemini-2.5-pro"))
    return getattr(_thread_local, key)

# ─────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────
CHUNK_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
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
                    "type": {"type": "STRING"},
                    "status": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "labels": {"type": "STRING"},
                    "scientific_significance": {"type": "STRING"},
                    "render_target": {"type": "STRING"},
                    "latex": {"type": "STRING"},
                    "placeholder_note": {"type": "STRING"}
                },
                "required": ["type"]
            },
            "sub_type": {"type": "STRING"},
            "difficulty": {"type": "STRING"},
            "source_example_id": {"type": "STRING"},
            "steps": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "step": {"type": "INTEGER"},
                        "action_display": {"type": "STRING"},
                        "justification": {"type": "STRING"}
                    }
                }
            },
            "final_answer": {"type": "STRING"},
            "final_answer_display": {"type": "STRING"},
            "exam_tip": {
                "type": "OBJECT",
                "properties": {
                    "question_pattern": {"type": "STRING"},
                    "skill_tested": {"type": "STRING"},
                    "distractor": {"type": "STRING"},
                    "why_distractor_works": {"type": "STRING"}
                }
            },
            "prerequisites_display": {"type": "ARRAY", "items": {"type": "STRING"}},
            "coverage_summary": {
                "type": "OBJECT",
                "properties": {
                    "definitions": {"type": "INTEGER"},
                    "formulas": {"type": "INTEGER"},
                    "laws": {"type": "INTEGER"},
                    "chemical_equations": {"type": "INTEGER"},
                    "properties": {"type": "INTEGER"},
                    "experiments": {"type": "INTEGER"},
                    "worked_examples": {"type": "INTEGER"}
                }
            },
            "next_modules": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["type", "slide_title", "script", "script_display", "display_bullets", "layout_config", "tts", "visual"]
    }
}

# ═════════════════════════════════════════
# SHARED RULES
# ═════════════════════════════════════════
SHARED_RULES = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TTS SCRIPT RULES — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"script" fields are read by a Kannada TTS engine.
TTS reads EVERY character literally. So:

  (a+b) -> TTS says "open bracket a plus b close bracket" = DISASTER
  H2O   -> TTS says "H two O" = OK if spoken naturally
  $x$   -> TTS says "dollar x dollar" = DISASTER

BANNED CHARACTERS in script:
  ( ) [ ] { } $ \\ ^ _ = + - * / % < > | ~ @

BANNED WORDS in script:
  "sub", "superscript", "subscript", "open bracket", "close bracket",
  "open parenthesis", "close parenthesis", "fraction", "over",
  "backslash", "LaTeX", "caret", "underscore"

SPOKEN SCIENCE — how to write science notation in script:
  H₂O -> "H two O" or "ನೀರು"
  CO₂ -> "C O two" or "ಇಂಗಾಲದ ಡೈಆಕ್ಸೈಡ್"
  H₂SO₄ -> "H two S O four" or "ಸಲ್ಫ್ಯೂರಿಕ್ ಆಮ್ಲ"
  NaOH -> "sodium hydroxide" or "ಸೋಡಿಯಂ ಹೈಡ್ರಾಕ್ಸೈಡ್"
  F = ma -> "F equals m times a"
  v² = u² + 2as -> "v squared equals u squared plus two a s"
  = -> "equals" or "is",  + -> "plus",  - -> "minus"
  → (reaction arrow) -> "gives" or "ಉತ್ಪನ್ನಗೊಳ್ಳುತ್ತದೆ"
  °C -> "degrees Celsius" or "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್"

DUAL FIELD RULE:
  "script"          -> for EARS (TTS). Zero symbols. Pure spoken Kannada.
  "script_display"  -> for EYES (screen). Same content WITH formulas/symbols.

display_bullets: MUST be array of STRINGS. Script MUST mention each bullet point.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE DESIGN — PRODUCTION QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Think YouTube education channel quality. Each slide = one screen in a video.

RULES:
- Each slide should feel SPACIOUS. Max 3-4 bullets per slide.
- If content is too much for one slide, SPLIT into multiple slides.
- Don't cram. White space is good.
- One idea per slide. One concept per slide.
- A worked example with 6 steps = 2-3 slides, not one giant slide.
- Teacher narration should be natural and unhurried.
- Quality over compression. More slides is better than crowded slides.

AUDIO-VISUAL SYNC:
- Each slide gets ONE audio (the "script" field).
- Everything on screen is visible from the start. Audio plays over it.
- Slide appears -> all content visible -> audio plays -> next slide.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE LAYOUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"title_hero"         — Big centered title. For: intro, section headers.
"definition_spotlight" — Term + definition + example. For: definitions.
"split_comparison"   — Left vs right. For: correct vs incorrect, comparing.
"formula_showcase"   — Formula large + explanation. For: key formulas, equations.
"step_walkthrough"   — 2-3 steps shown, teacher narrates. For: solutions, procedures.
"problem_setup"      — Problem statement + given/find. For: example intro.
"visual_explain"     — Text left, visual right. For: diagrams, apparatus, structures.
"visual_full"        — Visual dominant, caption below. For: important diagrams.
"bullet_list"        — Clean bullets, all visible. For: properties, classifications.
"key_takeaway"       — Highlighted result box. For: answers, exam tips.
"recap_grid"         — Summary grid. For: module ending.

layout_config schema:
{
  "layout": "<from above>",
  "template": "<matching template>",
  "text_zone": "<left|center|full|bottom>",
  "visual_zone": "<right|center|none>",
  "transition": {"in": "fade", "out": "fade", "duration_ms": 400}
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TTS CONFIG — same for ALL slides
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every slide uses:
  {"read_field": "script", "language": "kn-IN", "sync_mode": "chunk"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL AIDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIAGRAM (image to be added later):
  {"type": "diagram", "status": "needs_image", "description": "...", "labels": "...",
   "scientific_significance": "...", "placeholder_note": "Image to be generated"}

TABLE (render directly):
  {"type": "table", "status": "ready", "render_target": "html_table",
   "headers": [], "rows": [], "caption": ""}

FORMULA / EQUATION BOX:
  {"type": "formula_box", "status": "ready", "render_target": "katex",
   "latex": "", "description": ""}

No visual: {"type": "none"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOW & CONTINUITY — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All slides play back-to-back as ONE continuous video lesson.
The INTRO slide handles the greeting. Everything after flows naturally.

BANNED PHRASES in script (for ALL slides EXCEPT intro):
  "ನಮಸ್ಕಾರ", "ಸ್ವಾಗತ", "Hello", "Welcome"
  "ಈ ಪಾಠದಲ್ಲಿ ನಾವು...", "ಶುರು ಮಾಡೋಣ"

USE THESE TRANSITIONS:
  "ಈಗ ನಾವು..." / "ಮುಂದಿನ ವಿಷಯಕ್ಕೆ ಬರೋಣ." / "ಒಂದು ಉದಾಹರಣೆ ನೋಡೋಣ."
  "ಮುಂದುವರಿಸೋಣ." / "ಇಲ್ಲಿ ಗಮನಿಸಿ..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- GPSTR/HSTR teacher recruitment exam prep — Kannada medium
- All explanations in Kannada. Science terms: Kannada (English)
- For definitions: TEACH with a simple example, don't just state
- For laws: explain what it MEANS practically with real-world examples
- For chemical equations: read the equation in words, explain reactants/products
- For experiments: describe the setup, procedure, and what we observe
- For worked examples: include ACTUAL calculations with numbers
- For diagrams: describe what students should visualize
"""

# ═════════════════════════════════════════
# CHUNK TYPE PROMPTS
# ═════════════════════════════════════════

INTRO_PROMPT = """
GENERATE: INTRO SLIDES

Create the opening slides for a GPSTR science video lesson module.
Make it engaging — like a great YouTube education channel.

Use "title_hero" layout for opening. If prerequisites exist, add a "bullet_list" slide.

- script: Warm teacher welcome. What is this topic? Why does it matter?
  Real-life connection. Make students WANT to learn this. 50+ words.
- display_bullets: key things they'll learn

IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects.
"""

DEFINITION_PROMPT = """
GENERATE: DEFINITION SLIDES

Teach ONE definition in a GPSTR science video lesson.
Don't just state it — TEACH it. Make it click.

Ideas for slides:
- "definition_spotlight": The formal definition stated clearly
- "visual_explain": A concrete example or diagram
- "split_comparison": Example vs counter-example

script: Teach in simple Kannada. Break it down. 60+ words per slide.
IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "definition"
"""

LAW_PROMPT = """
GENERATE: SCIENTIFIC LAW SLIDES

Teach ONE scientific law/principle in a GPSTR science video lesson.
Make students understand the WHY, not just memorize the statement.

Ideas for slides:
- "definition_spotlight": The law statement in simple words
- "formula_showcase": The formula if it has one
- "visual_explain": Diagram showing the law in action
- "step_walkthrough": Real-world application or numerical example
- "key_takeaway": When it applies, exceptions, exam tips

script: 60+ words per slide. Explain practically.
IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "concept_explanation"
"""

CHEMICAL_EQUATION_PROMPT = """
GENERATE: CHEMICAL EQUATION SLIDES

Teach ONE chemical equation/reaction in a GPSTR science video lesson.
Make it visual and memorable.

Ideas for slides:
- "formula_showcase": The balanced equation displayed large
- "visual_explain": Diagram of the reaction process
- "bullet_list": Reactants, products, conditions, type of reaction

script: READ the equation in spoken words. Explain what happens.
  NOT "$2H_2 + O_2$" but "two molecules of hydrogen react with one molecule of oxygen"
IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "concept_explanation"
"""

EXPERIMENT_PROMPT = """
GENERATE: EXPERIMENT/ACTIVITY SLIDES

Teach ONE experiment/activity in a GPSTR science video lesson.
Walk students through as if they're in the lab.

Ideas for slides:
- "visual_explain": Apparatus setup diagram
- "step_walkthrough": Procedure steps
- "key_takeaway": Observation and conclusion

script: Describe vividly. "ನಾವು ಒಂದು ಬೀಕರ್ ತೆಗೆದುಕೊಳ್ಳೋಣ..." 60+ words per slide.
IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "concept_explanation"
"""

FORMULA_PROMPT = """
GENERATE: FORMULA SLIDES

Teach ONE scientific formula in a GPSTR science video lesson.

Ideas for slides:
- "formula_showcase": Formula displayed large. Read in spoken form.
  Explain each symbol. When do you use this?
- "step_walkthrough": Quick substitution example with actual numbers
- "key_takeaway": Common mistakes, when NOT to use

script: 60+ words per slide. visual: formula_box with KaTeX.
IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "concept_explanation"
"""

PROPERTY_PROMPT = """
GENERATE: PROPERTY/CLASSIFICATION SLIDES

Teach a set of properties or a classification in a GPSTR science video lesson.
Max 3 items per slide — keep it spacious.

Use "bullet_list" or "visual_explain". For each: state it, give example, explain WHY.

script: 60+ words per slide.
IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "concept_explanation"
"""

WORKED_EXAMPLE_PROMPT = """
GENERATE: WORKED EXAMPLE SLIDES

Solve ONE worked example in a GPSTR science video lesson.
Take your time. Don't rush. Multiple slides are BETTER than cramped.

APPROACH:
- "problem_setup" slide: problem displayed clearly. Read it. Identify given/find.
- "step_walkthrough" slides: Max 3 steps per slide.
  Each step needs ACTUAL numbers and calculations in script.
  NOT "substitute values" but "mass is 50 kg, acceleration is 10 m per second squared,
  so force equals 50 times 10 which gives us 500 Newtons"
- "key_takeaway" slide: Final answer. Common trap if relevant.

IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. type: "worked_example"
Include source_example_id on every slide.
"""

RECAP_PROMPT = """
GENERATE: RECAP SLIDE

Final slide of this module. Quick, punchy summary.

1 slide only: "recap_grid" layout
- script: 40+ words. Quick summary of key concepts, the ONE formula to remember, one GPSTR tip.
- display_bullets: 3-4 most important takeaways
- coverage_summary: counts of what was covered
- next_modules: from connects_to
- visual: {"type": "none"}

IMPORTANT: display_bullets MUST match what script says.

{shared_rules}

OUTPUT: Return a JSON ARRAY with ONE chunk object. type: "recap"
"""

# ═════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════

def format_prompt(template):
    return template.replace("{shared_rules}", SHARED_RULES)

def call_gemini(prompt):
    project_idx = getattr(_thread_project_idx, "idx", 0)
    m = get_model(project_idx)
    response = m.generate_content(
        [prompt],
        generation_config=GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=CHUNK_RESPONSE_SCHEMA,
        ),
    )
    result = json.loads(response.text)
    if isinstance(result, dict):
        result = [result]
    return result

def safe_call(prompt, label):
    max_retries = 15
    for attempt in range(max_retries):
        try:
            return call_gemini(prompt), None
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

def save_chunk_file(chunk_dir, seq, chunk_type, index, data):
    fname = f"{seq:03d}_{chunk_type}_{index}.json"
    (chunk_dir / fname).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return fname

# ═════════════════════════════════════════
# PROCESS ONE MODULE
# ═════════════════════════════════════════
def process_module(chapter_code, module_file, project_idx=0):
    _thread_project_idx.idx = project_idx
    module_path = MODULES_FOLDER / chapter_code / module_file
    module_id = module_file.replace(".json", "")

    chunk_dir = CHUNKS_FOLDER / chapter_code / module_id
    meta_path = chunk_dir / "_meta.json"

    # Skip if already done
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not old_meta.get("errors"):
                return "skipped"
        except Exception:
            return "skipped"

    chunk_dir.mkdir(parents=True, exist_ok=True)

    module_data = json.loads(module_path.read_text(encoding="utf-8"))

    theory = module_data.get("theory", {})
    definitions = theory.get("definitions", [])
    formulas = theory.get("formulas", [])
    laws = theory.get("laws", [])
    chem_equations = theory.get("chemical_equations", [])
    properties = theory.get("properties", [])
    classifications = theory.get("classifications", [])
    experiments = theory.get("experiments", [])
    visual_aids = theory.get("visual_aids", [])
    examples = module_data.get("worked_examples", [])
    exam_intel = module_data.get("exam_intelligence", {})

    prop_batches = [properties[i:i+3] for i in range(0, len(properties), 3)] if properties else []

    api_calls = (1 + len(definitions) + len(laws) + len(chem_equations) +
                 len(formulas) + len(prop_batches) + len(classifications) +
                 len(experiments) + len(examples) + 1)

    print(f"  Chunking: {module_id} ({api_calls} calls: "
          f"D:{len(definitions)} L:{len(laws)} CE:{len(chem_equations)} "
          f"F:{len(formulas)} P:{len(prop_batches)} Cl:{len(classifications)} "
          f"Exp:{len(experiments)} E:{len(examples)})")

    chunk_seq = 1
    chunk_order = []
    errors = []
    total_slides = 0

    module_context = f"""
MODULE CONTEXT:
module_id: {module_id}
module_title: {module_data.get('module_title', '')}
chapter_title: {module_data.get('chapter_title', '')}
class: {module_data.get('class', '')}
domain: {module_data.get('domain', '')}
concept_summary: {theory.get('concept_summary', '')}
exam_intelligence: {json.dumps(exam_intel, ensure_ascii=False)}
visual_aids_in_module: {json.dumps(visual_aids, indent=2, ensure_ascii=False)}
"""

    def do_call(prompt, label, chunk_type, index, extra_fields=None):
        nonlocal chunk_seq, total_slides
        expected_file = f"{chunk_seq:03d}_{chunk_type}_{index}.json"
        if (chunk_dir / expected_file).exists():
            existing = [f for f in os.listdir(chunk_dir)
                       if f.endswith(".json") and f != "_meta.json"
                       and (f == expected_file or f"_{chunk_type}_cont_{index}_" in f)]
            for ef in sorted(existing):
                chunk_order.append(ef)
                chunk_seq += 1
                total_slides += 1
            return

        result, err = safe_call(prompt, label)
        if result:
            for slide_idx, slide in enumerate(result):
                slide["chunk_id"] = f"{module_id}_chunk_{chunk_seq:03d}"
                if extra_fields:
                    slide.update(extra_fields)
                if slide_idx == 0:
                    fname = save_chunk_file(chunk_dir, chunk_seq, chunk_type, index, slide)
                else:
                    fname = save_chunk_file(chunk_dir, chunk_seq, f"{chunk_type}_cont", f"{index}_{slide_idx}", slide)
                chunk_order.append(fname)
                chunk_seq += 1
                total_slides += 1
        elif err:
            errors.append(err)
            print(f"    FAILED {label}: {err[:150]}")

    # --- 1. INTRO ---
    prompt = format_prompt(INTRO_PROMPT) + module_context
    prompt += f"\nprerequisites: {json.dumps(module_data.get('prerequisites', []), ensure_ascii=False)}"
    do_call(prompt, f"{module_id}_intro", "intro", 0)

    # --- 2. DEFINITIONS ---
    for i, defn in enumerate(definitions):
        prompt = format_prompt(DEFINITION_PROMPT) + module_context
        prompt += f"\nDefinition {i+1} of {len(definitions)}"
        prompt += f"\n\nDEFINITION TO TEACH:\n{json.dumps(defn, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_def_{i}", "definition", i)

    # --- 3. LAWS ---
    for i, law in enumerate(laws):
        prompt = format_prompt(LAW_PROMPT) + module_context
        prompt += f"\nLaw {i+1} of {len(laws)}"
        prompt += f"\n\nLAW TO TEACH:\n{json.dumps(law, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_law_{i}", "law", i)

    # --- 4. CHEMICAL EQUATIONS ---
    for i, eq in enumerate(chem_equations):
        prompt = format_prompt(CHEMICAL_EQUATION_PROMPT) + module_context
        prompt += f"\nEquation {i+1} of {len(chem_equations)}"
        prompt += f"\n\nCHEMICAL EQUATION TO TEACH:\n{json.dumps(eq, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_chem_{i}", "chemical_eq", i)

    # --- 5. FORMULAS ---
    for i, formula in enumerate(formulas):
        prompt = format_prompt(FORMULA_PROMPT) + module_context
        prompt += f"\nFormula {i+1} of {len(formulas)}"
        prompt += f"\n\nFORMULA TO TEACH:\n{json.dumps(formula, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_formula_{i}", "formula", i)

    # --- 6. PROPERTIES ---
    for i, batch in enumerate(prop_batches):
        prompt = format_prompt(PROPERTY_PROMPT) + module_context
        prompt += f"\nProperty batch {i+1} of {len(prop_batches)}"
        prompt += f"\n\nPROPERTIES TO TEACH:\n{json.dumps(batch, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_prop_{i}", "property", i)

    # --- 7. CLASSIFICATIONS ---
    for i, cls in enumerate(classifications):
        prompt = format_prompt(PROPERTY_PROMPT) + module_context
        prompt += f"\nClassification {i+1} of {len(classifications)}"
        prompt += f"\n\nCLASSIFICATION TO TEACH:\n{json.dumps(cls, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_class_{i}", "classification", i)

    # --- 8. EXPERIMENTS ---
    for i, exp in enumerate(experiments):
        prompt = format_prompt(EXPERIMENT_PROMPT) + module_context
        prompt += f"\nExperiment {i+1} of {len(experiments)}"
        prompt += f"\n\nEXPERIMENT TO TEACH:\n{json.dumps(exp, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_exp_{i}", "experiment", i)

    # --- 9. WORKED EXAMPLES ---
    for i, example in enumerate(examples):
        prompt = format_prompt(WORKED_EXAMPLE_PROMPT) + module_context
        prompt += f"\nExample {i+1} of {len(examples)}"
        prompt += f"\n\nWORKED EXAMPLE TO SOLVE:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
        extra = {"source_example_id": example.get("example_id", f"{module_id}_example_{i+1}")}
        do_call(prompt, f"{module_id}_ex_{i}", "example", i, extra_fields=extra)

    # --- 10. RECAP ---
    prompt = format_prompt(RECAP_PROMPT) + module_context
    prompt += f"\nCovered: {len(definitions)} defs, {len(laws)} laws, {len(chem_equations)} chem eqs, "
    prompt += f"{len(formulas)} formulas, {len(properties)} properties, "
    prompt += f"{len(experiments)} experiments, {len(examples)} examples"
    prompt += f"\nconnects_to: {json.dumps(exam_intel.get('connects_to', []), ensure_ascii=False)}"
    do_call(prompt, f"{module_id}_recap", "recap", 0)

    # --- SAVE META ---
    meta = {
        "module_id": module_id,
        "module_title": module_data.get("module_title", ""),
        "chapter_title": module_data.get("chapter_title", ""),
        "class": module_data.get("class", ""),
        "chapter": module_data.get("chapter", ""),
        "domain": module_data.get("domain", ""),
        "target_exam": "GPSTR",
        "total_slides": total_slides,
        "chunk_order": chunk_order,
        "errors": errors
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    status = f"  OK {module_id}: {total_slides} slides"
    if errors:
        status += f" ({len(errors)} errors)"
    print(status)
    return "ok" if not errors else "partial"

# ═════════════════════════════════════════
# FIND PENDING MODULES
# ═════════════════════════════════════════
def find_pending():
    pending = []
    if not MODULES_FOLDER.exists():
        return pending
    for chapter_dir in sorted(os.listdir(MODULES_FOLDER)):
        chapter_path = MODULES_FOLDER / chapter_dir
        if not chapter_path.is_dir():
            continue
        for f in sorted(os.listdir(chapter_path)):
            if not f.endswith(".json") or f == "validation.json":
                continue
            module_id = f.replace(".json", "")
            meta_path = CHUNKS_FOLDER / chapter_dir / module_id / "_meta.json"
            if meta_path.exists():
                try:
                    old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not old_meta.get("errors"):
                        continue
                except Exception:
                    continue
            pending.append((chapter_dir, f))
    return pending

# ═════════════════════════════════════════
# WORKER WRAPPERS
# ═════════════════════════════════════════
def process_module_with_project(args):
    project_idx, chapter_code, module_file = args
    return process_module(chapter_code, module_file, project_idx)

def run_project_batch(project_idx, modules_list):
    """Run a batch of modules using one project's credentials with THREADS_PER_PROJECT threads."""
    p = PROJECTS[project_idx]
    tasks = [(project_idx, ch, mf) for ch, mf in modules_list]
    print(f"\n[P{project_idx}] {p['project_id']} — {len(tasks)} modules, {THREADS_PER_PROJECT} threads")

    results = {"ok": 0, "error": 0, "skipped": 0, "partial": 0}
    with ThreadPoolExecutor(max_workers=THREADS_PER_PROJECT) as executor:
        futures = {executor.submit(process_module_with_project, t): t for t in tasks}
        for future in as_completed(futures):
            t = futures[future]
            try:
                status = future.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                print(f"  [P{project_idx}] Unexpected error on {t[1]}/{t[2]}: {e}")
                results["error"] += 1

    print(f"[P{project_idx}] {p['project_id']} — DONE (ok:{results['ok']} partial:{results['partial']} err:{results['error']})")
    return results

def process_batch(pending):
    """Split pending modules across all projects, run each project in parallel."""
    n_projects = len(PROJECTS)

    # Round-robin split across projects
    batches = [[] for _ in range(n_projects)]
    for i, item in enumerate(pending):
        batches[i % n_projects].append(item)

    combined = {"ok": 0, "error": 0, "skipped": 0, "partial": 0}

    with ThreadPoolExecutor(max_workers=n_projects) as project_executor:
        futures = []
        for i, batch in enumerate(batches):
            if batch:
                futures.append(project_executor.submit(run_project_batch, i, batch))
        for f in futures:
            r = f.result()
            for k in combined:
                combined[k] += r.get(k, 0)

    return combined

# ═════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════
def print_summary():
    total_modules = 0
    total_slides = 0
    total_errors = 0
    chapter_stats = {}

    for mc in sorted(os.listdir(CHUNKS_FOLDER)):
        mc_path = CHUNKS_FOLDER / mc
        if not mc_path.is_dir():
            continue
        ch_modules = 0
        ch_slides = 0
        ch_errors = 0
        for mod_dir in sorted(os.listdir(mc_path)):
            meta_path = mc_path / mod_dir / "_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ch_modules += 1
                    ch_slides += meta.get("total_slides", 0)
                    ch_errors += len(meta.get("errors", []))
                except Exception:
                    continue
        if ch_modules:
            chapter_stats[mc] = {"modules": ch_modules, "slides": ch_slides, "errors": ch_errors}
            total_modules += ch_modules
            total_slides += ch_slides
            total_errors += ch_errors

    print(f"\n{'='*60}")
    print(f"CHUNK GENERATION SCIENCE — SUMMARY")
    print(f"{'='*60}")
    print(f"Total chapters: {len(chapter_stats)}")
    print(f"Total modules: {total_modules}")
    print(f"Total slides: {total_slides}")
    print(f"Avg slides/module: {total_slides/max(total_modules,1):.1f}")
    print(f"Total errors: {total_errors}")

    print(f"\nPer chapter:")
    for code, info in sorted(chapter_stats.items()):
        err = f", {info['errors']} err" if info['errors'] else ""
        print(f"  {code}: {info['modules']} modules, {info['slides']} slides{err}")

    summary = {"total_chapters": len(chapter_stats), "total_modules": total_modules,
               "total_slides": total_slides, "total_errors": total_errors, "chapters": chapter_stats}
    summary_path = CHUNKS_FOLDER / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")

# ═════════════════════════════════════════
# MAIN — POLLING LOOP
# ═════════════════════════════════════════
def main():
    n_projects = len(PROJECTS)
    print(f"\n{'='*60}")
    print(f"SCIENCE CHUNK GENERATION — {n_projects} projects")
    print(f"{'='*60}")
    for i, p in enumerate(PROJECTS):
        print(f"  Project {i}: {p['project_id']}")
    print(f"Threads per project: {THREADS_PER_PROJECT}")
    print(f"Max concurrent: {n_projects * THREADS_PER_PROJECT}")
    print(f"Watching: {MODULES_FOLDER}")
    print(f"Output: {CHUNKS_FOLDER}\n")

    total_processed = 0
    empty_polls = 0

    while empty_polls < MAX_POLLS:
        pending = find_pending()

        if DEBUG:
            pending = pending[:1]

        if not pending:
            empty_polls += 1
            if empty_polls == 1:
                print(f"\nNo pending modules. Polling every {POLL_INTERVAL}s...")
            elif empty_polls % 10 == 0:
                print(f"  Still waiting... ({empty_polls} polls)")
            time.sleep(POLL_INTERVAL)
            continue

        empty_polls = 0
        print(f"\nFound {len(pending)} pending modules")

        results = process_batch(pending)
        total_processed += results.get("ok", 0) + results.get("partial", 0)

        print(f"\nBatch done — ok:{results.get('ok',0)} partial:{results.get('partial',0)} error:{results.get('error',0)}")
        print(f"Total processed so far: {total_processed}")

        if DEBUG:
            break

    print_summary()
    print("\nDone!")

if __name__ == "__main__":
    main()
