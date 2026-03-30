import os
import json
import time
import threading
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor
import asyncio

# -----------------------------
# CONFIG
# -----------------------------
LOCATION = "us-central1"
CREDENTIALS_FOLDER = "claude_works/credentials"
MODULES_FOLDER = "claude_works/modules"
CHUNKS_FOLDER = "claude_works/chunks_structured"

MAX_WORKERS_PER_PROJECT = 4
DEBUG = True

# -----------------------------
# MULTI-PROJECT INIT
# -----------------------------
PROJECTS = []
for cred_file in sorted(os.listdir(CREDENTIALS_FOLDER)):
    if not cred_file.endswith(".json"):
        continue
    cred_path = os.path.join(CREDENTIALS_FOLDER, cred_file)
    creds = service_account.Credentials.from_service_account_file(cred_path)
    project_id = json.load(open(cred_path))["project_id"]
    PROJECTS.append({"project_id": project_id, "credentials": creds, "cred_file": cred_file})
    print(f"  Loaded project: {project_id}")

print(f"Total projects: {len(PROJECTS)}")

# Thread-local storage for model per project
_thread_local = threading.local()

def get_model_for_project(project_idx):
    """Get or create a model for the given project index (cached per thread)."""
    key = f"model_{project_idx}"
    if not hasattr(_thread_local, key):
        p = PROJECTS[project_idx]
        vertexai.init(project=p["project_id"], location=LOCATION, credentials=p["credentials"])
        setattr(_thread_local, key, GenerativeModel("gemini-2.5-pro"))
    return getattr(_thread_local, key)

# Which project the current thread should use
_thread_project_idx = threading.local()

os.makedirs(CHUNKS_FOLDER, exist_ok=True)

# -----------------------------
# STRUCTURED OUTPUT SCHEMA
# -----------------------------
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
                    "mathematical_significance": {"type": "STRING"},
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
                    "theorems": {"type": "INTEGER"},
                    "properties": {"type": "INTEGER"},
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
  a^2   -> TTS says "a caret two" = DISASTER
  $x$   -> TTS says "dollar x dollar" = DISASTER

BANNED CHARACTERS in script:
  ( ) [ ] { } $ \\ ^ _ = + - * / % < > | ~ @

BANNED WORDS in script:
  "sub", "superscript", "subscript", "open bracket", "close bracket",
  "open parenthesis", "close parenthesis", "fraction", "over",
  "backslash", "LaTeX", "caret", "underscore"

SPOKEN MATH — how to write math in script:
  a_1 -> "a one",  a_n -> "a n",  S_n -> "S n"  (NEVER say "sub")
  a^2 -> "a squared",  a^3 -> "a cubed",  x^n -> "x to the power n"
  (a+b)^2 -> "a plus b, whole squared"
  (n-1)d -> "n minus one, times d"  (NEVER say "open bracket")
  a/b -> "a by b",  n/2 -> "n by two"  (NEVER say "fraction" or "over")
  sqrt(x) -> "square root of x"
  = -> "equals" or "is",  + -> "plus",  - -> "minus"

FULL FORMULA EXAMPLES:
  $a_n = a + (n-1)d$  ->  "a n equals a plus n minus one, times d"
  $S_n = \\frac{n}{2}(a + l)$  ->  "S n equals n by two, times a plus l"

DUAL FIELD RULE:
  "script"          -> for EARS (TTS). Zero symbols. Pure spoken Kannada.
  "script_display"  -> for EYES (screen). Same content WITH LaTeX ($...$).

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

Pick the BEST layout for the content:

"title_hero"         — Big centered title. For: intro, section headers.
"definition_spotlight" — Term + definition + example. For: definitions.
"split_comparison"   — Left vs right. For: correct vs incorrect, comparing.
"formula_showcase"   — Formula large + explanation. For: key formulas.
"step_walkthrough"   — 2-3 steps shown, teacher narrates through them. For: solutions, proofs.
"problem_setup"      — Problem statement + given/find. For: example intro.
"visual_explain"     — Text left, visual right. For: diagrams, tables.
"visual_full"        — Visual dominant, caption below. For: important diagrams.
"bullet_list"        — Clean bullets, all visible. For: properties, rules.
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

ONE audio per slide. "script" is the only narration source.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL AIDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIAGRAM (image to be added later):
  {"type": "diagram", "status": "needs_image", "description": "...", "labels": "...",
   "mathematical_significance": "...", "placeholder_note": "Image to be generated"}

TABLE (render directly):
  {"type": "table", "status": "ready", "render_target": "html_table",
   "headers": [], "rows": [], "caption": ""}

FORMULA BOX:
  {"type": "formula_box", "status": "ready", "render_target": "katex",
   "latex": "", "description": ""}

No visual: {"type": "none"}

When visual exists (even needs_image), use visual layout variant.
Script CAN reference figures — image WILL be there in final video.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM TIP (optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Include ONLY when genuinely useful for GPSTR. Don't force.
  {"question_pattern": "MCQ|2-mark|5-mark", "skill_tested": "",
   "distractor": "", "why_distractor_works": ""}

Skip on: intro, recap, basic examples.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOW & CONTINUITY — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All slides play back-to-back as ONE continuous video lesson.
Think of it as a 15-minute YouTube class — NOT 20 separate clips.
The INTRO slide handles the greeting and lesson overview.
Everything after that must flow like a teacher continuing to talk.

BANNED PHRASES in script (for ALL slides EXCEPT intro):
  Greetings:
    "ನಮಸ್ಕಾರ", "ನಮಸ್ಕಾರ ವಿದ್ಯಾರ್ಥಿಗಳೇ", "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೇ"
    "ಸ್ವಾಗತ", "ಈ ವೀಡಿಯೋಗೆ ಸ್ವಾಗತ"
    "Hello", "Welcome", "Hi students"

  Lesson-start phrases (already said in intro):
    "ಈ ಪಾಠದಲ್ಲಿ ನಾವು..." (In this lesson we...)
    "ಈ ಅಧ್ಯಾಯದಲ್ಲಿ..." (In this chapter...)
    "ಈ ಮಾಡ್ಯೂಲ್ ನಲ್ಲಿ..." (In this module...)
    "ಶುರು ಮಾಡೋಣ" / "ಪ್ರಾರಂಭಿಸೋಣ" (Let's begin/start)
    "ಸಿದ್ಧರಿದ್ದೀರಾ?" (Are you ready?)

  Sign-off phrases (only for recap):
    "ಮುಂದಿನ ಪಾಠದಲ್ಲಿ ಭೇಟಿಯಾಗೋಣ" (See you next lesson)
    "ಇಷ್ಟೇ ಈ ಪಾಠ" (That's it for this lesson)

USE THESE TRANSITIONS INSTEAD:
  Between topics:
    "ಈಗ ನಾವು..." (Now we...)
    "ಮುಂದಿನ ವಿಷಯಕ್ಕೆ ಬರೋಣ." (Let's move to the next topic.)
    "ಇನ್ನೊಂದು ಪ್ರಮುಖ ವಿಷಯವನ್ನು ನೋಡೋಣ." (Let's see another important topic.)

  Starting a definition/formula/theorem:
    "ಇಲ್ಲಿ ಗಮನಿಸಿ..." (Notice here...)
    "ಈ ಪರಿಕಲ್ಪನೆಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳೋಣ." (Let's understand this concept.)
    "ಮೊದಲು ವ್ಯಾಖ್ಯೆಯನ್ನು ನೋಡೋಣ." (First let's look at the definition.)

  Starting an example:
    "ಒಂದು ಉದಾಹರಣೆ ನೋಡೋಣ." (Let's see an example.)
    "ಈ ಸಮಸ್ಯೆಯನ್ನು ಬಿಡಿಸೋಣ." (Let's solve this problem.)

  Continuing from previous slide:
    "ಮುಂದುವರಿಸೋಣ." (Let's continue.)
    "ಈಗ ಮುಂದಿನ ಹಂತಕ್ಕೆ ಬರೋಣ." (Now let's come to the next step.)

REMEMBER: Each slide's script is a CONTINUATION of the previous one.
The student just heard the previous slide 2 seconds ago. Don't re-introduce.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- GPSTR/HSTR teacher recruitment exam prep — Kannada medium
- All explanations in Kannada. Math terms: Kannada (English)
- For definitions: TEACH with a simple numerical example, don't just state
- For formulas: read in spoken math, explain each symbol, give quick example with real numbers
- For worked examples: include ACTUAL calculations with numbers in script
  NOT "substitute values" but "substitute a equals 3 and d equals 5, we get 3 plus 9 times 5 which is 48"
- For theorems: explain what it MEANS practically, not just formal statement
- For properties: give a numerical example proving each property
"""

# ═════════════════════════════════════════
# CHUNK TYPE PROMPTS
# ═════════════════════════════════════════

INTRO_PROMPT = """
GENERATE: INTRO SLIDES

You are creating the opening slides for a GPSTR math video lesson module.
Make it engaging — like a great YouTube education channel.

Create slides for the module intro. Use as many slides as needed.

Use "title_hero" layout for the opening. If prerequisites exist, add a separate
"bullet_list" slide for what students should already know.

- script: Warm teacher welcome. What is this concept? Why does it matter?
  Real-life connection if possible. Make students WANT to learn this. 50+ words.
- display_bullets: key things they'll learn
IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
Student reads bullets on screen while hearing script — they must match.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects. Even if just 1 slide, return as array.
Each chunk needs: chunk_id, type ("intro"), layout_config, tts, slide_title, script,
script_display (optional), display_bullets, visual.
"""

DEFINITION_PROMPT = """
GENERATE: DEFINITION SLIDES

You are teaching ONE definition in a GPSTR math video lesson.
Don't just state it — TEACH it. Make it click.

Create slides to TEACH this definition. Use as many slides as the content needs.
Don't cram everything onto one slide. Each slide should have one clear idea.

Ideas for slides (use what fits, skip what doesn't):
- "definition_spotlight": The formal definition stated clearly
- "step_walkthrough" or "visual_explain": A SIMPLE concrete example
  "For example, look at the numbers 3, 7, 11, 15.
  Here we add 4 each time. So this is an arithmetic progression."
- "split_comparison": Example vs counter-example side by side
  "THIS is an AP... but THIS is NOT, because..."

script: Teach the definition in simple Kannada. Break it down. 60+ words per slide.
IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
Student reads bullets on screen while hearing script — they must match.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects.
type: "definition" for all slides in this group.
"""

FORMULA_PROMPT = """
GENERATE: FORMULA SLIDES

You are teaching ONE formula in a GPSTR math video lesson.
Make students understand it, not just memorize it.

Create slides to TEACH this formula. Use as many slides as needed.
A simple formula may need fewer slides. A complex one with derivation needs more.

Ideas for slides (use what fits):
- "formula_showcase": The formula displayed large and clear. Read it in spoken math.
  Explain what each symbol means. When do you use this?
- "step_walkthrough": Derivation steps (if derivation exists in the module data)
- "step_walkthrough": Quick substitution example with actual numbers
  "if first term is 3 and difference is 5, then tenth term is..."
- "key_takeaway": When to use vs when NOT to use, common mistakes

script: 60+ words per slide. visual: formula_box with KaTeX for the formula slide.
IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
Student reads bullets on screen while hearing script — they must match.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects.
type: "concept_explanation" or "formula_derivation" as appropriate.
"""

THEOREM_PROMPT = """
GENERATE: THEOREM SLIDES

You are teaching ONE theorem in a GPSTR math video lesson.
Make it understandable, not intimidating.

Create slides to TEACH this theorem. Use as many slides as the theorem needs.
A theorem with a long proof needs more slides. A simple one needs fewer.

Ideas for slides (use what fits):
- "definition_spotlight": Theorem statement in simple words FIRST, then formal
- "step_walkthrough": Proof steps (if proof exists). Split across multiple slides
  if proof is long — never cram a long proof onto one slide.
- "split_comparison" or "step_walkthrough": Converse, or a concrete example
- "key_takeaway": Special cases, common errors, GPSTR exam tips

script: 60+ words per slide.
IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
Student reads bullets on screen while hearing script — they must match.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects.
type: "definition" for statement slides, "concept_explanation" for examples.
"""

PROPERTY_PROMPT = """
GENERATE: PROPERTY/RULE SLIDES

You are teaching a set of properties in a GPSTR math video lesson.
Each property should feel clear, not rushed.

Create slides to explain these properties. Use as many slides as needed.
Max 3 properties per slide — keep it spacious and clear.

Use "bullet_list" or "visual_explain" (if table/number_line available).
For each property: state it, give a quick numerical example, explain WHY.
"The square of an even number is always even. For example, 4 squared is 16."

script: 60+ words per slide.
IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
Student reads bullets on screen while hearing script — they must match.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects.
type: "concept_explanation" for all.
"""

WORKED_EXAMPLE_PROMPT = """
GENERATE: WORKED EXAMPLE SLIDES

You are solving ONE worked example in a GPSTR math video lesson.
This is the MOST IMPORTANT content — students learn by watching solutions.
Take your time. Don't rush. Multiple slides are BETTER than one cramped slide.

Create slides for this worked example. Use as many slides as the solution needs.
Don't rush. Don't cram. Each slide should feel clean and readable.

APPROACH:
- Start with a "problem_setup" slide: problem displayed clearly.
  script: Read the problem. Identify what's given and what to find.
  "Let us read this problem carefully. We are given that... We need to find..."
  Don't start solving yet.

- Then "step_walkthrough" slides for the solution.
  Max 3 steps per slide. If example has 6 steps, use 2-3 solution slides.
  Each step needs:
  * action_spoken: Teacher narrating with ACTUAL numbers (15+ words).
    NOT "substitute values" but:
    "Now we substitute. a equals 3 and d equals 5. So a ten equals 3 plus 9 times 5.
     9 times 5 is 45. 3 plus 45 gives us 48."
  * action_display: LaTeX version
  * justification: which formula/property
  All steps visible at once on slide. ONE script narrates through all of them.

- End with a "key_takeaway" slide for the final answer.
  script: State the answer clearly. Mention common trap if relevant.

IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
Student reads bullets on screen while hearing script — they must match.

{shared_rules}

OUTPUT: Return a JSON ARRAY of chunk objects.
type: "worked_example" for all slides in this example.
Include source_example_id on every slide from the same example.
"""

RECAP_PROMPT = """
GENERATE: RECAP SLIDE

Generate the FINAL slide of this module. Quick, punchy summary.

1 slide only: "recap_grid" layout
- slide_title: recap title
- script: 40+ words. Quick summary:
  * Name the key concepts covered
  * The ONE formula to remember
  * One GPSTR tip
  Short and energetic. "That's it for this module! Remember..."
- display_bullets: 3-4 most important takeaways
IMPORTANT: Whatever you put in display_bullets, you MUST say the same thing in script.
- coverage_summary: counts of what was covered
- next_modules: from connects_to
- visual: {"type": "none"}
- NO exam_tip on recap

{shared_rules}

OUTPUT: Return a JSON ARRAY with ONE chunk object.
type: "recap"
"""

# ═════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════

def format_prompt(template):
    return template.replace("{shared_rules}", SHARED_RULES)

def call_gemini(prompt):
    from vertexai.generative_models import GenerationConfig
    project_idx = getattr(_thread_project_idx, "idx", 0)
    m = get_model_for_project(project_idx)
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
    max_retries = 20
    for attempt in range(max_retries):
        try:
            return call_gemini(prompt), None
        except json.JSONDecodeError:
            wait = 10 * (attempt + 1)
            print(f"      JSON parse error, retrying in {wait}s... ({attempt+1}/{max_retries}) [{label}]")
            time.sleep(wait)
            continue
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s... exponential backoff
                print(f"      429 rate limit, waiting {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                time.sleep(wait)
                continue
            elif "500" in err or "502" in err or "503" in err or "504" in err or "INTERNAL" in err or "UNAVAILABLE" in err:
                wait = 30 * (attempt + 1)
                print(f"      Server error ({err[:80]}), waiting {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                time.sleep(wait)
                continue
            elif "DEADLINE_EXCEEDED" in err or "timeout" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"      Timeout, waiting {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                time.sleep(wait)
                continue
            else:
                # Unknown error — still retry a few times before giving up
                if attempt < 5:
                    wait = 15 * (attempt + 1)
                    print(f"      Error: {err[:100]}, retrying in {wait}s... ({attempt+1}/{max_retries}) [{label}]")
                    time.sleep(wait)
                    continue
                return None, f"{label}: {e}"
    return None, f"{label}: max retries ({max_retries}) exhausted"

# ═════════════════════════════════════════
# PROCESS ONE MODULE
# ═════════════════════════════════════════
def save_chunk_file(chunk_dir, seq, chunk_type, index, data):
    """Save one chunk: 001_intro_0.json, 002_definition_0.json, etc."""
    fname = f"{seq:03d}_{chunk_type}_{index}.json"
    with open(os.path.join(chunk_dir, fname), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return fname

def process_module(merge_code, module_file):

    module_path = os.path.join(MODULES_FOLDER, merge_code, module_file)
    module_id = os.path.splitext(module_file)[0]

    chunk_dir = os.path.join(CHUNKS_FOLDER, merge_code, module_id)
    meta_path = os.path.join(chunk_dir, "_meta.json")

    # Skip if already done
    if os.path.exists(meta_path):
        try:
            old_meta = json.load(open(meta_path, encoding="utf-8"))
            if not old_meta.get("errors"):
                print(f"  >> Skip: {module_id}")
                return "skipped"
            else:
                print(f"  Retrying {len(old_meta['errors'])} failed calls: {module_id}")
                # Don't delete — will skip existing chunk files below
        except Exception:
            print(f"  >> Skip: {module_id}")
            return "skipped"

    os.makedirs(chunk_dir, exist_ok=True)

    with open(module_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)

    theory = module_data.get("theory", {})
    definitions = theory.get("definitions", [])
    formulas = theory.get("formulas", [])
    theorems = theory.get("theorems", [])
    properties = theory.get("properties", [])
    visual_aids = theory.get("visual_aids", [])
    examples = module_data.get("worked_examples", [])
    exam_intel = module_data.get("exam_intelligence", {})

    prop_batches = [properties[i:i+3] for i in range(0, len(properties), 3)] if properties else []

    api_calls = 1 + len(definitions) + len(formulas) + len(theorems) + len(prop_batches) + len(examples) + 1

    print(f"  Chunking: {module_id} ({api_calls} calls: "
          f"D:{len(definitions)} F:{len(formulas)} T:{len(theorems)} "
          f"P:{len(prop_batches)} E:{len(examples)} V:{len(visual_aids)})")

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
        # Check if first file from this call already exists — skip if so
        expected_file = f"{chunk_seq:03d}_{chunk_type}_{index}.json"
        if os.path.exists(os.path.join(chunk_dir, expected_file)):
            # Count existing files from this call
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

    # --- 3. THEOREMS ---
    for i, thm in enumerate(theorems):
        prompt = format_prompt(THEOREM_PROMPT) + module_context
        prompt += f"\nTheorem {i+1} of {len(theorems)}"
        prompt += f"\n\nTHEOREM TO TEACH:\n{json.dumps(thm, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_thm_{i}", "theorem", i)

    # --- 4. PROPERTIES ---
    for i, batch in enumerate(prop_batches):
        prompt = format_prompt(PROPERTY_PROMPT) + module_context
        prompt += f"\nProperty batch {i+1} of {len(prop_batches)}"
        prompt += f"\n\nPROPERTIES TO TEACH:\n{json.dumps(batch, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_prop_{i}", "property", i)

    # --- 5. FORMULAS ---
    for i, formula in enumerate(formulas):
        prompt = format_prompt(FORMULA_PROMPT) + module_context
        prompt += f"\nFormula {i+1} of {len(formulas)}"
        prompt += f"\n\nFORMULA TO TEACH:\n{json.dumps(formula, indent=2, ensure_ascii=False)}"
        do_call(prompt, f"{module_id}_formula_{i}", "formula", i)

    # --- 6. WORKED EXAMPLES ---
    for i, example in enumerate(examples):
        prompt = format_prompt(WORKED_EXAMPLE_PROMPT) + module_context
        prompt += f"\nExample {i+1} of {len(examples)}"
        prompt += f"\n\nWORKED EXAMPLE TO SOLVE:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
        extra = {"source_example_id": example.get("example_id", f"{module_id}_example_{i+1}")}
        do_call(prompt, f"{module_id}_ex_{i}", "example", i, extra_fields=extra)

    # --- 7. RECAP ---
    prompt = format_prompt(RECAP_PROMPT) + module_context
    prompt += f"\nCovered: {len(definitions)} defs, {len(formulas)} formulas, "
    prompt += f"{len(theorems)} theorems, {len(properties)} properties, {len(examples)} examples"
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

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    status = f"  OK {module_id}: {total_slides} slides in {len(chunk_order)} files"
    if errors:
        status += f" ({len(errors)} errors)"
    print(status)

    return "ok" if not errors else "partial"

# ═════════════════════════════════════════
# PROCESS ONE CHAPTER
# ═════════════════════════════════════════
def process_chapter(merge_code):
    module_dir = os.path.join(MODULES_FOLDER, merge_code)
    if not os.path.isdir(module_dir):
        return
    module_files = sorted([f for f in os.listdir(module_dir) if f.endswith(".json")])
    if not module_files:
        return
    print(f"\n== {merge_code} ({len(module_files)} modules) ==")
    for mf in module_files:
        process_module(merge_code, mf)

# ═════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════
def generate_summary():
    print("\n" + "=" * 60)
    print("CHUNK GENERATION SUMMARY")
    print("=" * 60)

    total_modules = 0
    total_slides = 0
    total_errors = 0
    chapter_stats = {}

    for mc in sorted(os.listdir(CHUNKS_FOLDER)):
        mc_path = os.path.join(CHUNKS_FOLDER, mc)
        if not os.path.isdir(mc_path):
            continue
        ch_modules = 0
        ch_slides = 0
        ch_errors = 0
        for mod_dir in sorted(os.listdir(mc_path)):
            meta_path = os.path.join(mc_path, mod_dir, "_meta.json")
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path, encoding="utf-8"))
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

    print(f"\nTotal chapters: {len(chapter_stats)}")
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
    summary_path = os.path.join(CHUNKS_FOLDER, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

# ═════════════════════════════════════════
# WORKER — runs chapters for one project
# ═════════════════════════════════════════
def process_module_with_project(args):
    """Process a single module using the assigned project credentials."""
    project_idx, merge_code, module_file = args
    _thread_project_idx.idx = project_idx
    process_module(merge_code, module_file)

def run_project_batch(project_idx, chapters):
    """Run a batch of chapters using one project's credentials, with threads."""
    p = PROJECTS[project_idx]
    print(f"\n[Project {project_idx}] {p['project_id']} — {len(chapters)} chapters, {MAX_WORKERS_PER_PROJECT} threads")

    # Collect all (merge_code, module_file) pairs for this project's chapters
    all_tasks = []
    for code in chapters:
        module_dir = os.path.join(MODULES_FOLDER, code)
        if not os.path.isdir(module_dir):
            continue
        module_files = sorted([f for f in os.listdir(module_dir) if f.endswith(".json")])
        print(f"  [P{project_idx}] {code}: {len(module_files)} modules")
        for mf in module_files:
            all_tasks.append((project_idx, code, mf))

    print(f"  [P{project_idx}] Total modules: {len(all_tasks)}")

    # Run modules in parallel within this project
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PROJECT) as executor:
        executor.map(process_module_with_project, all_tasks)

    print(f"\n[Project {project_idx}] {p['project_id']} — DONE")

# ═════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════
def main():
    merge_codes = sorted([d for d in os.listdir(MODULES_FOLDER) if os.path.isdir(os.path.join(MODULES_FOLDER, d))])

    if DEBUG:
        # Debug: 1 merge code (all its modules), 1 project
        merge_codes = merge_codes[:1]
        mc = merge_codes[0]
        mc_dir = os.path.join(MODULES_FOLDER, mc)
        module_files = sorted([f for f in os.listdir(mc_dir) if f.endswith(".json")])
        print(f"{'='*60}")
        print(f"DEBUG MODE — 1 merge code: {mc} ({len(module_files)} modules)")
        print(f"{'='*60}")
        print(f"  Project: {PROJECTS[0]['project_id']}\n")
        _thread_project_idx.idx = 0
        for mf in module_files:
            process_module(mc, mf)
        generate_summary()
        print("\nDone")
        return

    total_modules = sum(len([f for f in os.listdir(os.path.join(MODULES_FOLDER, mc)) if f.endswith(".json")]) for mc in merge_codes)

    n_projects = len(PROJECTS)

    print(f"{'='*60}")
    print(f"CHUNK GENERATION — structured schema, {n_projects} projects")
    print(f"{'='*60}")
    print(f"{len(merge_codes)} chapters, {total_modules} modules\n")

    # Split chapters evenly across projects
    batches = [[] for _ in range(n_projects)]
    for i, code in enumerate(merge_codes):
        batches[i % n_projects].append(code)

    for i, batch in enumerate(batches):
        print(f"  Project {i} ({PROJECTS[i]['project_id']}): {len(batch)} chapters — {batch}")

    # Run each project in its own thread pool
    with ThreadPoolExecutor(max_workers=n_projects) as executor:
        futures = []
        for i, batch in enumerate(batches):
            if batch:
                futures.append(executor.submit(run_project_batch, i, batch))
        for f in futures:
            f.result()  # wait for all to finish

    generate_summary()
    print("\nDone")

if __name__ == "__main__":
    main()
