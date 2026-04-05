"""
Generate HTML study notes from master data files.
- Reads master data from claude_works/master_data/{MERGE_CODE}.txt
- Gemini generates ONLY the body content (no CSS, no head)
- Python wraps it in a fixed HTML template with CSS, KaTeX, fonts
- Saves to claude_works/notes/{MERGE_CODE}.html
- Multi-project parallel, skip-if-done, retry on failure
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
MASTER_DATA_FOLDER = Path("claude_works/master_data")
NOTES_FOLDER       = Path("claude_works/notes")
CREDENTIALS_FOLDER = Path("claude_works/credentials")
PROMPT_FILE        = Path("prompts_v2/generate_notes.txt")

LOCATION = "us-central1"
MAX_WORKERS_PER_PROJECT = 2
DEBUG = False

NOTES_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────
# LOAD PROMPT
# ─────────────────────────────────────
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

# ─────────────────────────────────────
# HTML TEMPLATE — fixed shell, Gemini fills the body
# ─────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="kn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — GPSTR ಗಣಿತ ಟಿಪ್ಪಣಿ</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[
    {{left:'$$',right:'$$',display:true}},
    {{left:'\\\\[',right:'\\\\]',display:true}},
    {{left:'$',right:'$',display:false}},
    {{left:'\\\\(',right:'\\\\)',display:false}}
  ]}});"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Kannada:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --primary: #2563eb;
  --primary-light: #eff6ff;
  --green: #16a34a;
  --green-light: #f0fdf4;
  --purple: #7c3aed;
  --purple-light: #faf5ff;
  --orange: #ea580c;
  --orange-light: #fff7ed;
  --red: #dc2626;
  --red-light: #fef2f2;
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-500: #6b7280;
  --gray-700: #374151;
  --gray-900: #111827;
  --radius: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
  color: var(--gray-900);
  background: var(--gray-50);
  line-height: 1.7;
  font-size: 16px;
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
}}
.page-header {{
  text-align: center;
  padding: 40px 24px;
  background: linear-gradient(135deg, var(--primary), #1d4ed8);
  color: white;
  border-radius: var(--radius);
  margin-bottom: 32px;
}}
.page-header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.page-header h1 small {{ font-size: 18px; font-weight: 400; opacity: 0.85; }}
.domain-tag {{
  display: inline-block;
  background: rgba(255,255,255,0.2);
  padding: 4px 16px;
  border-radius: 20px;
  font-size: 14px;
  margin-top: 8px;
}}
.weightage-badge {{
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  margin-left: 8px;
}}
.weightage-high {{ background: #fecaca; color: #991b1b; }}
.weightage-medium {{ background: #fed7aa; color: #9a3412; }}
.weightage-low {{ background: #d1fae5; color: #065f46; }}
.toc {{
  background: white;
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 32px;
  box-shadow: var(--shadow);
}}
.toc h2 {{ font-size: 18px; margin-bottom: 12px; color: var(--gray-700); }}
.toc ol {{ padding-left: 20px; }}
.toc li {{ margin-bottom: 6px; }}
.toc a {{ color: var(--primary); text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
.formula-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}}
.formula-card {{
  background: white;
  border-left: 4px solid var(--green);
  border-radius: var(--radius);
  padding: 16px;
  box-shadow: var(--shadow);
}}
.formula-card .fname {{ font-size: 13px; color: var(--gray-500); margin-bottom: 8px; }}
.formula-card .flatex {{ font-size: 18px; text-align: center; padding: 8px 0; }}
.concept {{
  background: white;
  border-radius: var(--radius);
  padding: 32px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}}
.concept-header {{
  border-bottom: 2px solid var(--gray-200);
  padding-bottom: 16px;
  margin-bottom: 24px;
}}
.concept-number {{
  font-size: 13px;
  color: var(--primary);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
}}
.concept-title {{ font-size: 22px; font-weight: 700; margin: 4px 0 12px; }}
.summary {{ color: var(--gray-700); font-size: 15px; line-height: 1.8; }}
.prereq-tags {{ margin-top: 12px; }}
.prereq-tag {{
  display: inline-block;
  background: var(--gray-100);
  color: var(--gray-700);
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
  margin-right: 6px;
}}
.section-title {{
  font-size: 16px;
  font-weight: 600;
  color: var(--gray-700);
  margin: 28px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--gray-200);
}}
.definition-box {{
  background: var(--primary-light);
  border-left: 4px solid var(--primary);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 16px 20px;
  margin-bottom: 12px;
}}
.definition-box .term {{ font-weight: 600; margin-bottom: 4px; }}
.definition-box .def-text {{ color: var(--gray-700); font-size: 15px; }}
.formula-box {{
  background: var(--green-light);
  border-left: 4px solid var(--green);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 16px 20px;
  margin-bottom: 12px;
}}
.formula-box .formula-name {{ font-weight: 600; color: var(--green); font-size: 14px; }}
.formula-box .formula-latex {{ font-size: 20px; text-align: center; padding: 12px 0; }}
.formula-box .formula-meta {{ font-size: 13px; color: var(--gray-500); margin-top: 4px; }}
.formula-box .formula-meta strong {{ color: var(--gray-700); }}
details.derivation {{
  margin-top: 8px;
  background: var(--gray-50);
  border-radius: 8px;
  padding: 8px 12px;
}}
details.derivation summary {{
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: var(--green);
}}
details.derivation .deriv-step {{
  padding: 6px 0;
  border-bottom: 1px solid var(--gray-100);
  font-size: 14px;
}}
.theorem-box {{
  background: var(--purple-light);
  border-left: 4px solid var(--purple);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 16px 20px;
  margin-bottom: 12px;
}}
.theorem-box .theorem-name {{ font-weight: 600; color: var(--purple); font-size: 14px; }}
.theorem-box .theorem-statement {{
  font-style: italic;
  padding: 8px 0;
  border-left: 2px solid var(--purple);
  padding-left: 12px;
  margin: 8px 0;
  color: var(--gray-700);
}}
details.proof {{
  margin-top: 8px;
  background: var(--gray-50);
  border-radius: 8px;
  padding: 8px 12px;
}}
details.proof summary {{
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: var(--purple);
}}
.property-list {{ padding-left: 20px; margin-bottom: 16px; }}
.property-list li {{ margin-bottom: 8px; color: var(--gray-700); }}
.example {{
  border: 1px solid var(--gray-200);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
}}
.example-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }}
.difficulty-badge {{
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}}
.diff-basic {{ background: #d1fae5; color: #065f46; }}
.diff-intermediate {{ background: #fef9c3; color: #854d0e; }}
.diff-advanced {{ background: #fecaca; color: #991b1b; }}
.problem-box {{
  background: var(--gray-100);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
  font-weight: 500;
}}
.step {{
  display: flex;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--gray-100);
}}
.step-number {{
  width: 28px;
  height: 28px;
  background: var(--primary);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
  flex-shrink: 0;
}}
.step-content {{ flex: 1; }}
.step-action {{ margin-bottom: 4px; }}
.step-calc {{ font-size: 17px; padding: 4px 0; }}
.step-justify {{ font-size: 12px; color: var(--gray-500); }}
.answer-box {{
  background: var(--green-light);
  border: 1px solid var(--green);
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 12px;
  font-weight: 600;
}}
.trap-warning {{
  background: var(--orange-light);
  border-left: 4px solid var(--orange);
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  margin-top: 12px;
  font-size: 14px;
}}
.exam-tip {{
  background: var(--primary-light);
  border: 1px solid var(--primary);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-top: 20px;
}}
.exam-tip h4 {{ color: var(--primary); font-size: 14px; margin-bottom: 8px; }}
.exam-tip .tip-row {{ font-size: 14px; margin-bottom: 4px; color: var(--gray-700); }}
.exam-tip .tip-row strong {{ color: var(--gray-900); }}
.diagram-placeholder {{
  border: 2px dashed var(--gray-200);
  border-radius: var(--radius);
  padding: 24px;
  text-align: center;
  margin: 16px 0;
  background: var(--gray-50);
}}
.diagram-placeholder .diagram-label {{
  font-size: 16px;
  font-weight: 600;
  color: var(--gray-500);
  margin-bottom: 8px;
}}
.page-footer {{
  text-align: center;
  padding: 24px;
  color: var(--gray-500);
  font-size: 13px;
  border-top: 1px solid var(--gray-200);
  margin-top: 32px;
}}
@media print {{
  body {{ max-width: 100%; padding: 0; background: white; }}
  .concept {{ box-shadow: none; border: 1px solid var(--gray-200); break-inside: avoid; }}
  .page-header {{ background: var(--primary); -webkit-print-color-adjust: exact; }}
}}
</style>
</head>
<body>

{body_content}

</body>
</html>"""

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
# GEMINI CALL WITH RETRY
# ─────────────────────────────────────
def call_gemini(prompt, project_idx):
    model = get_model(project_idx)
    response = model.generate_content(
        [prompt],
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=65536,
        ),
    )
    return response.text

def safe_call(prompt, label, project_idx):
    max_retries = 15
    for attempt in range(max_retries):
        try:
            return call_gemini(prompt, project_idx), None
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
# CLEAN BODY CONTENT
# ─────────────────────────────────────
def extract_body(response_text):
    """Extract just the body HTML from Gemini response."""
    text = response_text.strip()

    # Strip markdown fences
    if text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # If Gemini returned a full HTML doc despite instructions, extract body
    body_start = text.lower().find("<body")
    if body_start >= 0:
        # Find the end of <body...>
        body_tag_end = text.find(">", body_start) + 1
        body_close = text.lower().rfind("</body>")
        if body_close > body_tag_end:
            text = text[body_tag_end:body_close].strip()

    # If it starts with <!DOCTYPE or <html, strip everything before first real content
    if text.lower().startswith("<!doctype") or text.lower().startswith("<html"):
        # Find first div
        first_div = text.find("<div")
        if first_div > 0:
            text = text[first_div:]
        # Strip trailing </html>
        html_end = text.lower().rfind("</html>")
        if html_end > 0:
            text = text[:html_end].strip()

    # Strip any <style> blocks that leaked in
    while "<style" in text.lower():
        style_start = text.lower().find("<style")
        style_end = text.lower().find("</style>")
        if style_start >= 0 and style_end > style_start:
            text = text[:style_start] + text[style_end + 8:]
        else:
            break

    # Strip <link> and <script> tags that leaked in
    import re
    text = re.sub(r'<link\s[^>]*>', '', text)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)

    return text.strip()

# ─────────────────────────────────────
# EXTRACT TOPIC NAME FROM MASTER DATA
# ─────────────────────────────────────
def get_topic_name(master_text):
    """Extract topic name from master data for HTML title."""
    for line in master_text.split("\n"):
        line = line.strip()
        if line.startswith("topic:"):
            return line[6:].strip()
    return "GPSTR ಗಣಿತ"

# ─────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────
def validate_body(body_content):
    """Basic checks on generated body content."""
    warnings = []
    if '<div class="page-header"' not in body_content:
        warnings.append("Missing page-header")
    if '<div class="concept"' not in body_content and 'class="concept"' not in body_content:
        warnings.append("No concept sections found")
    if '<div class="page-footer"' not in body_content:
        warnings.append("Missing page-footer")

    dollar_count = body_content.count("$$")
    inline_count = body_content.count("$") - dollar_count * 2
    if dollar_count == 0 and inline_count == 0:
        warnings.append("No LaTeX formulas found")

    return warnings

# ─────────────────────────────────────
# PROCESS ONE MASTER DATA FILE
# ─────────────────────────────────────
def process_master_data(merge_code, project_idx):
    master_path = MASTER_DATA_FOLDER / f"{merge_code}.txt"
    html_path = NOTES_FOLDER / f"{merge_code}.html"
    meta_path = NOTES_FOLDER / f"{merge_code}_meta.json"

    # Skip if already done
    if html_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "ok":
                print(f"  >> Skip (done): {merge_code}")
                return "skipped"
        except Exception:
            pass

    # Read master data
    master_text = master_path.read_text(encoding="utf-8")
    topic_name = get_topic_name(master_text)
    concept_count = master_text.count("[ಮೂಲ_ಅವಧಾರಣೆ_")
    print(f"  Generating: {merge_code} — {topic_name} ({concept_count} concepts)")

    # Build prompt — just the lean prompt + master data
    prompt = SYSTEM_PROMPT + "\n\n"
    prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    prompt += "MASTER DATA INPUT:\n"
    prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    prompt += master_text

    # Call Gemini
    result, err = safe_call(prompt, merge_code, project_idx)

    if err:
        meta = {
            "merge_code": merge_code,
            "status": "error",
            "error": err,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  FAIL {merge_code}: {err[:80]}")
        return "error"

    # Extract body content
    body_content = extract_body(result)

    # Validate
    warnings = validate_body(body_content)

    # Wrap in template
    full_html = HTML_TEMPLATE.format(
        title=topic_name,
        body_content=body_content
    )

    # Save
    html_path.write_text(full_html, encoding="utf-8")

    meta = {
        "merge_code": merge_code,
        "topic": topic_name,
        "status": "ok",
        "concepts": concept_count,
        "body_size": len(body_content),
        "html_size": len(full_html),
        "warnings": warnings,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    warn_str = f" ({len(warnings)} warnings)" if warnings else ""
    print(f"  OK {merge_code}: {concept_count} concepts, body={len(body_content)//1024}KB{warn_str}")
    if warnings:
        for w in warnings:
            print(f"    ⚠ {w}")

    return "ok"

# ─────────────────────────────────────
# WORKER
# ─────────────────────────────────────
def process_with_project(args):
    project_idx, merge_code = args
    _thread_project_idx.idx = project_idx
    return process_master_data(merge_code, project_idx)

def run_project_batch(project_idx, merge_codes):
    p = PROJECTS[project_idx]
    print(f"\n[Project {project_idx}] {p['project_id']} — {len(merge_codes)} topics")
    tasks = [(project_idx, mc) for mc in merge_codes]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PROJECT) as executor:
        executor.map(process_with_project, tasks)
    print(f"\n[Project {project_idx}] DONE")

# ─────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────
def generate_summary():
    print("\n" + "=" * 60)
    print("NOTES GENERATION SUMMARY")
    print("=" * 60)

    total = 0
    ok = 0
    errors = 0
    total_concepts = 0
    total_size = 0
    total_warnings = 0
    results = {}

    for f in sorted(NOTES_FOLDER.glob("*_meta.json")):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            mc = meta.get("merge_code", f.stem.replace("_meta", ""))
            total += 1
            if meta.get("status") == "ok":
                ok += 1
                total_concepts += meta.get("concepts", 0)
                total_size += meta.get("html_size", 0)
                total_warnings += len(meta.get("warnings", []))
                results[mc] = {
                    "status": "ok",
                    "topic": meta.get("topic", ""),
                    "concepts": meta.get("concepts", 0),
                    "size_kb": meta.get("html_size", 0) // 1024,
                    "warnings": len(meta.get("warnings", []))
                }
            else:
                errors += 1
                results[mc] = {"status": "error", "error": meta.get("error", "")[:80]}
        except Exception:
            continue

    all_master = set(f.stem for f in MASTER_DATA_FOLDER.glob("*.txt"))
    done_codes = set(results.keys())
    pending = all_master - done_codes

    print(f"\nTotal master data: {len(all_master)}")
    print(f"Generated:  {ok} ok / {errors} error / {len(pending)} pending")
    print(f"Concepts:   {total_concepts}")
    print(f"Total size: {total_size//1024}KB ({total_size//1024//1024}MB)")
    print(f"Warnings:   {total_warnings}")

    if pending:
        print(f"\nPending: {', '.join(sorted(pending))}")

    print(f"\nPer topic:")
    for mc, info in sorted(results.items()):
        if info["status"] == "ok":
            warn = f", {info['warnings']}w" if info["warnings"] else ""
            print(f"  {mc:<20} OK  {info['concepts']}c  {info['size_kb']}KB{warn}  {info.get('topic','')}")
        else:
            print(f"  {mc:<20} ERR {info['error'][:50]}")

    summary = {
        "total": len(all_master), "ok": ok, "errors": errors, "pending": len(pending),
        "total_concepts": total_concepts, "total_size_bytes": total_size,
        "total_warnings": total_warnings, "topics": results
    }
    (NOTES_FOLDER / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {NOTES_FOLDER / 'generation_summary.json'}")

# ─────────────────────────────────────
# MAIN
# ─────────────────────────────────────
def main():
    all_files = sorted([f.stem for f in MASTER_DATA_FOLDER.glob("*.txt")])
    if not all_files:
        print("ERROR: No master data files found!")
        return

    pending = []
    for mc in all_files:
        meta_path = NOTES_FOLDER / f"{mc}_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") == "ok":
                    continue
            except Exception:
                pass
        pending.append(mc)

    if DEBUG:
        pending = pending[:1]

    n_projects = len(PROJECTS)

    print(f"{'='*60}")
    print(f"NOTES GENERATION")
    print(f"{'='*60}")
    print(f"Total master data: {len(all_files)}")
    print(f"Already done:      {len(all_files) - len(pending)}")
    print(f"To generate:       {len(pending)}")
    print(f"Projects:          {n_projects}")
    print()

    if not pending:
        print("Nothing to do!")
        generate_summary()
        return

    batches = [[] for _ in range(n_projects)]
    for i, mc in enumerate(pending):
        batches[i % n_projects].append(mc)

    for i, batch in enumerate(batches):
        print(f"  Project {i} ({PROJECTS[i]['project_id']}): {len(batch)} topics — {batch}")

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
