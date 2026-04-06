"""
generate_video_v3.py
---------------------
Full pipeline: Chunks → Slides → Video → AWS S3 Upload → MongoDB Save → Local Cleanup

Flow per chapter (strictly sequential):
  1. Check MongoDB — skip if already published
  2. Render HTML slides → PNG via Playwright
  3. FFmpeg: PNG + WAV → MP4 segments → concat → add intro/end
  4. Upload to S3 (retry 3x)
  5. Save metadata to MongoDB (retry 3x)
  6. Delete local MP4 + slides folder
  7. Only then move to next chapter

Error handling:
  - Retries 3 times on S3/MongoDB failures
  - After 3 failures, logs error and STOPS (does not continue to next chapter)
  - Error log: claude_works/video_pipeline_errors.log
  - Progress log: claude_works/video_pipeline_progress.log
  - On rerun, skips chapters already published in MongoDB

ENV vars needed (.env file):
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_REGION           (default: ap-south-1)
  AWS_S3_BUCKET        (default: gpstr-maths-videos)
  MONGODB_URI          (default: mongodb+srv://user:user@cluster0.rgocxdb.mongodb.net/gpstr-maths-db)
"""

import json
import asyncio
import subprocess
import multiprocessing
import wave
import os
import sys
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from playwright.async_api import async_playwright

FFMPEG_WORKERS = max(2, multiprocessing.cpu_count() - 1)  # parallel FFmpeg segments

# ==============================
# LOAD .env
# ==============================

def load_dotenv(path=".env"):
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_dotenv()

# ==============================
# CONFIG
# ==============================

CHUNKS_DIR  = Path("claude_works/chunks_structured")
AUDIO_DIR   = Path("claude_works/audio_v2")
OUTPUT_DIR  = Path("claude_works/videos_v2")
DATA_JSON   = Path("data.json")

ERROR_LOG   = Path("claude_works/video_pipeline_errors.log")
PROGRESS_LOG = Path("claude_works/video_pipeline_progress.log")

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries

INTRO_VIDEO = Path("intro_v0.mp4")
END_VIDEO   = Path("end_v0.mp4")

WIDTH, HEIGHT = 1920, 1080
FPS           = 30
DEBUG         = False

# AWS
AWS_ACCESS_KEY  = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION      = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET       = os.environ.get("AWS_S3_BUCKET", "gpstr-maths-videos")

# MongoDB
MONGODB_URI     = os.environ.get("MONGODB_URI", "mongodb+srv://user:user@cluster0.rgocxdb.mongodb.net/gpstr-maths-db")
MONGO_DB_NAME   = "gpstr-maths-db"
MONGO_COLLECTION = "videos"

print("CPU cores:", multiprocessing.cpu_count())


# ==============================
# LOGGING
# ==============================

def log_progress(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(merge_code, stage, error_msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] FAIL | {merge_code} | {stage} | {error_msg}"
    print(f"  ERROR: {line}")
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def retry_operation(func, merge_code, stage, *args, **kwargs):
    """Retry func up to MAX_RETRIES times. Returns result or raises after 3 failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            log_error(merge_code, f"{stage} (attempt {attempt}/{MAX_RETRIES})", str(e))
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise

# ==============================
# LOAD DATA.JSON (chapter metadata lookup)
# ==============================

def load_chapter_metadata():
    """Build merge_code -> chapter info lookup from data.json."""
    if not DATA_JSON.exists():
        return {}
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for domain, chapters in data.items():
        for ch in chapters:
            mc = ch["merge_code"]
            if mc not in lookup:
                lookup[mc] = {
                    "domain": domain,
                    "merge_code": mc,
                    "classes": [],
                    "chapters": [],
                    "chapter_name": ch["chapter_name"],
                }
            lookup[mc]["classes"].append(ch["class"])
            lookup[mc]["chapters"].append(ch["chapter_no"])
    # dedupe
    for v in lookup.values():
        v["classes"] = sorted(set(v["classes"]))
        v["chapters"] = sorted(set(v["chapters"]))
    return lookup

CHAPTER_META = load_chapter_metadata()

# ==============================
# AWS S3 UPLOAD
# ==============================

def get_s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

def upload_to_s3(local_path, s3_key):
    """Upload file to S3. Returns the public URL."""
    s3 = get_s3_client()
    s3.upload_file(
        str(local_path),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    return url


# ==============================
# MONGODB SAVE
# ==============================

def get_mongo_collection():
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

def save_to_mongodb(doc):
    """Upsert video document by merge_code."""
    col = get_mongo_collection()
    col.update_one(
        {"merge_code": doc["merge_code"]},
        {"$set": doc},
        upsert=True,
    )
    print(f"  MongoDB: saved {doc['merge_code']}")


# ==============================
# COLLECT MODULE INFO FOR A CHAPTER
# ==============================

def collect_module_info(merge_code):
    """Collect module names and concept info for a chapter."""
    chunk_base = CHUNKS_DIR / merge_code
    if not chunk_base.is_dir():
        return []
    modules = []
    for mod_dir in sorted(d for d in chunk_base.iterdir() if d.is_dir()):
        meta_path = mod_dir / "_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        modules.append({
            "module_id": meta.get("module_id", mod_dir.name),
            "module_title": meta.get("module_title", ""),
            "total_slides": meta.get("total_slides", 0),
        })
    return modules


# ==============================
# DESIGN TOKENS
# ==============================

BRAND   = "#00c6ff"
BG      = "#020202"
TEXT    = "#ffffff"
SUBTEXT = "#cfcfcf"

# ==============================
# GLOBAL STYLES
# ==============================

GLOBAL_STYLES = f"""
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {WIDTH}px; height: {HEIGHT}px;
    background: {BG};
    background-image: radial-gradient(circle at 75% 50%, #001a25 0%, {BG} 65%);
    font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
    color: {TEXT};
    display: flex; flex-direction: column;
    overflow: hidden;
  }}

  /* -- TOP HEADER -- */
  .top-header {{
    width: 100%; height: 54px;
    display: flex; align-items: center;
    padding: 0 60px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
  }}
  .brand {{
    font-size: 14px; font-weight: 600;
    letter-spacing: 3px; color: {BRAND};
    text-transform: uppercase;
  }}
  .header-meta {{
    margin-left: auto;
    font-size: 13px; color: rgba(255,255,255,0.25);
  }}

  /* -- CONTENT AREA -- */
  .content {{
    flex-grow: 1;
    padding: 50px 80px 50px 80px;
    display: flex; flex-direction: column;
    justify-content: center; overflow: hidden;
  }}

  /* -- COMMON ELEMENTS -- */
  .slide-title {{
    font-size: 52px; font-weight: 800;
    line-height: 1.2; margin-bottom: 18px;
    color: {TEXT};
  }}
  .accent-line {{
    width: 100px; height: 4px;
    background: {BRAND};
    margin-bottom: 44px; border-radius: 2px;
    box-shadow: 0 0 16px rgba(0,198,255,0.45);
  }}
  .two-col {{
    display: flex; gap: 60px;
    align-items: flex-start; flex-grow: 1;
  }}
  .col-text {{ flex: 1; }}
  .col-visual {{ flex: 0 0 520px; }}

  /* -- BULLETS -- */
  .bullet-list {{ list-style: none; }}
  .bullet-list li {{
    font-size: 34px; line-height: 1.55;
    margin-bottom: 24px; display: flex;
    align-items: flex-start; color: {SUBTEXT};
  }}
  .bullet-node {{
    width: 7px; height: 26px;
    background: {BRAND}; margin-top: 12px;
    margin-right: 24px; flex-shrink: 0;
    border-radius: 1px;
  }}

  /* -- STEP CARDS -- */
  .step-card {{
    border-radius: 10px; padding: 18px 24px;
    margin-bottom: 16px;
    border-left: 3px solid rgba(0,198,255,0.3);
    font-size: 30px; line-height: 1.5;
    color: {SUBTEXT};
    background: rgba(0,198,255,0.08);
  }}
  .step-card.active {{
    background: rgba(0,198,255,0.18);
    border-left-color: {BRAND};
    color: {TEXT};
  }}
  .step-num {{
    font-size: 20px; font-weight: 600;
    color: {BRAND}; margin-bottom: 4px;
  }}
  .step-justification {{
    font-size: 22px; color: rgba(255,255,255,0.45);
    margin-top: 6px; font-style: italic;
  }}

  /* -- DEFINITION CARD -- */
  .def-card {{
    background: rgba(0,198,255,0.07);
    border: 1px solid rgba(0,198,255,0.2);
    border-radius: 12px; padding: 32px 36px;
    margin-bottom: 24px;
  }}
  .def-term {{
    font-size: 34px; font-weight: 700;
    color: {BRAND}; margin-bottom: 12px;
  }}
  .def-body {{ font-size: 34px; line-height: 1.6; color: {SUBTEXT}; }}

  /* -- FORMULA BOX -- */
  .formula-box {{
    background: rgba(0,198,255,0.06);
    border: 1px solid rgba(0,198,255,0.25);
    border-radius: 12px; padding: 28px 36px;
    text-align: center; font-size: 42px;
    margin-bottom: 24px;
  }}
  .formula-label {{
    font-size: 22px; color: rgba(255,255,255,0.4);
    margin-bottom: 14px; text-transform: uppercase;
    letter-spacing: 2px;
  }}

  /* -- TABLE -- */
  table {{
    border-collapse: collapse; width: 100%;
    font-size: 26px;
    table-layout: fixed;
    border: 1px solid rgba(0,198,255,0.2);
    border-radius: 8px;
  }}
  th {{
    background: rgba(0,198,255,0.15);
    color: {BRAND}; padding: 14px 18px;
    text-align: left; font-weight: 600;
    border: 1px solid rgba(0,198,255,0.2);
    word-wrap: break-word; overflow-wrap: break-word;
  }}
  td {{
    padding: 12px 18px; color: {SUBTEXT};
    border: 1px solid rgba(255,255,255,0.08);
    word-wrap: break-word; overflow-wrap: break-word;
    vertical-align: top; line-height: 1.5;
  }}
  tr:nth-child(even) td {{ background: rgba(255,255,255,0.03); }}
  tr:nth-child(odd) td {{ background: rgba(0,0,0,0.15); }}

  /* -- EXAM TIP -- */
  .exam-tip {{
    background: rgba(255,200,0,0.07);
    border: 1px solid rgba(255,200,0,0.25);
    border-radius: 10px; padding: 20px 26px;
    font-size: 26px; margin-top: 24px;
  }}
  .exam-tip-label {{
    font-size: 18px; font-weight: 700;
    color: #ffc800; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px;
  }}

  /* -- PROBLEM BOX -- */
  .problem-box {{
    font-size: 30px; color: {SUBTEXT};
    background: rgba(255,255,255,0.04);
    padding: 18px 24px; border-radius: 8px;
    border-left: 3px solid rgba(0,198,255,0.3);
    margin-bottom: 28px; line-height: 1.6;
  }}

  /* -- ANSWER BOX -- */
  .answer-box {{
    margin-top: 20px; padding: 18px 24px;
    background: rgba(0,198,255,0.12);
    border-radius: 10px; font-size: 34px;
    font-weight: 700; color: {BRAND};
  }}

  /* -- RECAP GRID -- */
  .recap-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px; margin-top: 8px;
  }}
  .recap-cell {{
    background: rgba(0,198,255,0.07);
    border: 1px solid rgba(0,198,255,0.15);
    border-radius: 10px; padding: 24px;
    text-align: center;
  }}
  .recap-count {{
    font-size: 56px; font-weight: 800;
    color: {BRAND}; line-height: 1;
  }}
  .recap-label {{
    font-size: 22px; color: {SUBTEXT};
    margin-top: 8px;
  }}

  /* -- COMPARISON COLUMNS -- */
  .compare-col {{
    flex: 1; padding: 24px;
    border-radius: 12px;
  }}
  .compare-col.left {{
    background: rgba(0,198,255,0.06);
    border: 1px solid rgba(0,198,255,0.2);
  }}
  .compare-col.right {{
    background: rgba(255,100,100,0.06);
    border: 1px solid rgba(255,100,100,0.2);
  }}
  .compare-title {{
    font-size: 28px; font-weight: 700;
    margin-bottom: 16px;
  }}
  .compare-col.left .compare-title {{ color: {BRAND}; }}
  .compare-col.right .compare-title {{ color: #ff9999; }}

  /* -- KEY TAKEAWAY -- */
  .takeaway-box {{
    background: rgba(0,198,255,0.1);
    border: 2px solid rgba(0,198,255,0.3);
    border-radius: 14px; padding: 36px 40px;
    font-size: 36px; line-height: 1.6;
    color: {TEXT}; text-align: center;
  }}

  /* -- DIFFICULTY BADGE -- */
  .badge {{
    font-size: 20px; padding: 4px 16px;
    border-radius: 20px; margin-left: 16px;
    vertical-align: middle; display: inline-block;
  }}
  .badge-basic       {{ color: #44cc88; border: 1px solid #44cc88; }}
  .badge-intermediate {{ color: {BRAND}; border: 1px solid {BRAND}; }}
  .badge-advanced     {{ color: #ff6b6b; border: 1px solid #ff6b6b; }}

  /* -- DIAGRAM PLACEHOLDER -- */
  .diagram-placeholder {{
    border: 1px dashed rgba(0,198,255,0.3);
    border-radius: 10px; padding: 40px;
    text-align: center; color: rgba(255,255,255,0.3);
    font-size: 22px;
  }}
"""


# ==============================
# BASE HTML WRAPPER
# ==============================

def base_html(body, header_meta=""):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[
    {{left:'$$',right:'$$',display:true}},
    {{left:'$',right:'$',display:false}}
  ]}});">
</script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+Kannada:wght@400;700&display=swap" rel="stylesheet">
<style>{GLOBAL_STYLES}</style>
</head>
<body>
  <div class="top-header">
    <div class="brand">SRINIVAS IAS ACADEMY</div>
    <div class="header-meta">{header_meta}</div>
  </div>
  {body}
</body>
</html>"""


# ==============================
# HELPERS
# ==============================

def make_header_meta(chunk, meta):
    cls = meta.get("class", "")
    ch  = meta.get("chapter", "")
    return f"{cls}ನೇ ತರಗತಿ &nbsp;|&nbsp; ಅಧ್ಯಾಯ {ch}"


def make_bullets_html(bullets):
    if not bullets:
        return ""
    items = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets)
    return f'<ul class="bullet-list">{items}</ul>'


def make_visual_html(visual):
    if not visual or visual.get("type") == "none":
        return ""

    vtype = visual.get("type", "")

    if vtype == "table":
        headers = visual.get("headers") or []
        rows    = visual.get("rows") or []
        num_cols = len(headers) if headers else (len(rows[0]) if rows else 1)
        font_size = "22px" if num_cols > 4 else ("24px" if num_cols > 3 else "26px")
        th = "".join(f"<th>{h}</th>" for h in headers)
        tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        caption = visual.get("caption", "")
        cap_html = f'<div style="font-size:20px;color:rgba(255,255,255,0.4);margin-bottom:12px">{caption}</div>' if caption else ""
        return f"{cap_html}<table style='font-size:{font_size}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"

    if vtype == "formula_box":
        latex = visual.get("latex", "")
        desc  = visual.get("description", "")
        label = f'<div class="formula-label">{desc}</div>' if desc else ""
        return f'<div class="formula-box">{label}$${latex}$$</div>' if latex else ""

    if vtype == "diagram":
        desc = visual.get("description", "diagram")
        sig  = visual.get("mathematical_significance", "")
        return (f'<div class="diagram-placeholder">'
                f'[{desc}]'
                f'<br><span style="font-size:16px">{sig}</span>'
                f'</div>')

    return ""


def make_exam_tip_html(chunk):
    et = chunk.get("exam_tip")
    if not et:
        return ""
    pattern    = et.get("question_pattern", "")
    skill      = et.get("skill_tested", "")
    distractor = et.get("distractor", "")
    return f"""
    <div class="exam-tip">
      <div class="exam-tip-label">EXAM TIP - {pattern}</div>
      <div>{skill}</div>
      <div style="margin-top:6px;color:rgba(255,200,0,0.6);font-size:22px">{distractor}</div>
    </div>"""


def make_steps_html(steps):
    if not steps:
        return ""
    cards = ""
    for s in steps:
        action = s.get("action_display", "")
        justif = s.get("justification", "")
        jhtml  = f'<div class="step-justification">{justif}</div>' if justif else ""
        cards += f"""
        <div class="step-card">
          <div class="step-num">Step {s.get('step', '')}</div>
          <div>{action}</div>
          {jhtml}
        </div>"""
    return cards


def difficulty_badge(d):
    if not d:
        return ""
    cls = f"badge-{d}" if d in ("basic", "intermediate", "advanced") else "badge-intermediate"
    return f'<span class="badge {cls}">{d.upper()}</span>'


# ==============================
# LAYOUT RENDERERS
# ==============================

def layout_title_hero(chunk, meta):
    domain  = meta.get("domain", "")
    cls     = meta.get("class", "")
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    subtitle = f"""<div style="font-size:26px;color:{BRAND};letter-spacing:3px;
                   text-transform:uppercase;margin-bottom:20px">
                   {domain} - {cls}ನೇ ತರಗತಿ</div>""" if domain else ""
    vis_col = f'<div class="col-visual">{vis}</div>' if vis else ""
    if vis:
        inner = f'<div class="two-col"><div class="col-text">{bullets}</div>{vis_col}</div>'
    else:
        inner = bullets
    return f"""
    <div class="content" style="justify-content:center">
      {subtitle}
      <div class="slide-title" style="font-size:64px">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {inner}
    </div>"""


def layout_definition_spotlight(chunk, meta):
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    if vis:
        inner = f'<div class="two-col"><div class="col-text">{bullets}</div><div class="col-visual">{vis}</div></div>'
    else:
        inner = bullets
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {inner}
    </div>"""


def layout_formula_showcase(chunk, meta):
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    if not vis:
        v = chunk.get("visual", {})
        if v.get("latex"):
            vis = f'<div class="formula-box" style="font-size:48px">$${v["latex"]}$$</div>'
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      <div class="two-col">
        <div class="col-text">{bullets}</div>
        <div class="col-visual">{vis}</div>
      </div>
      {make_exam_tip_html(chunk)}
    </div>"""


def layout_step_walkthrough(chunk, meta):
    steps   = make_steps_html(chunk.get("steps", []))
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    text_content = steps if steps else bullets
    final_ans = chunk.get("final_answer_display") or chunk.get("final_answer", "")
    final_html = f'<div class="answer-box">{final_ans}</div>' if final_ans else ""
    if vis:
        inner = f"""<div class="two-col">
            <div class="col-text" style="overflow-y:auto;max-height:780px">{text_content}{final_html}</div>
            <div class="col-visual">{vis}</div></div>"""
    else:
        inner = f'<div style="overflow-y:auto;max-height:780px">{text_content}{final_html}</div>'
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {inner}
      {make_exam_tip_html(chunk)}
    </div>"""


def layout_problem_setup(chunk, meta):
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    diff    = difficulty_badge(chunk.get("difficulty", ""))
    problem = chunk.get("script_display", "")
    vis_col = f'<div class="col-visual">{vis}</div>' if vis else ""
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')} {diff}</div>
      <div class="accent-line"></div>
      <div class="problem-box">{problem if not bullets else ''}</div>
      {"<div class='two-col'><div class='col-text'>" + bullets + "</div>" + vis_col + "</div>" if vis else bullets}
    </div>"""


def layout_visual_explain(chunk, meta):
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      <div class="two-col">
        <div class="col-text">{bullets}</div>
        <div class="col-visual">{vis}</div>
      </div>
      {make_exam_tip_html(chunk)}
    </div>"""


def layout_visual_full(chunk, meta):
    vis     = make_visual_html(chunk.get("visual", {}))
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    return f"""
    <div class="content" style="align-items:center;text-align:center">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line" style="margin-left:auto;margin-right:auto"></div>
      <div style="max-width:900px;width:100%">{vis}</div>
      <div style="margin-top:28px;text-align:left;max-width:900px">{bullets}</div>
    </div>"""


def layout_bullet_list(chunk, meta):
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    vis     = make_visual_html(chunk.get("visual", {}))
    if vis:
        inner = f'<div class="two-col"><div class="col-text">{bullets}</div><div class="col-visual">{vis}</div></div>'
    else:
        inner = bullets
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {inner}
      {make_exam_tip_html(chunk)}
    </div>"""


def layout_key_takeaway(chunk, meta):
    final_ans = chunk.get("final_answer_display") or chunk.get("final_answer", "")
    bullets   = make_bullets_html(chunk.get("display_bullets", []))
    takeaway = f'<div class="takeaway-box">{final_ans}</div>' if final_ans else ""
    return f"""
    <div class="content" style="justify-content:center">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {takeaway}
      <div style="margin-top:32px">{bullets}</div>
      {make_exam_tip_html(chunk)}
    </div>"""


def layout_split_comparison(chunk, meta):
    bullets = chunk.get("display_bullets", [])
    mid = len(bullets) // 2 or 1
    left_items  = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets[:mid])
    right_items = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets[mid:])
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      <div class="two-col" style="gap:40px">
        <div class="compare-col left">
          <div class="compare-title">Correct</div>
          <ul class="bullet-list">{left_items}</ul>
        </div>
        <div class="compare-col right">
          <div class="compare-title">Incorrect</div>
          <ul class="bullet-list">{right_items}</ul>
        </div>
      </div>
    </div>"""


def layout_recap_grid(chunk, meta):
    cs = chunk.get("coverage_summary") or {}
    cells = [
        ("Definitions", cs.get("definitions", 0)),
        ("Formulas", cs.get("formulas", 0)),
        ("Theorems", cs.get("theorems", 0)),
        ("Properties", cs.get("properties", 0)),
        ("Worked Examples", cs.get("worked_examples", 0)),
    ]
    cells = [(k, v) for k, v in cells if v]
    grid = "".join(
        f'<div class="recap-cell"><div class="recap-count">{v}</div>'
        f'<div class="recap-label">{k}</div></div>'
        for k, v in cells
    )
    bullets = make_bullets_html(chunk.get("display_bullets", []))
    next_mods = chunk.get("next_modules") or []
    next_html = ""
    if next_mods:
        items = " / ".join(next_mods)
        next_html = f"""<div style="margin-top:28px;font-size:24px;color:rgba(255,255,255,0.35)">
          Next: <span style="color:{BRAND}">{items}</span></div>"""
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title', 'Recap')}</div>
      <div class="accent-line"></div>
      <div class="two-col">
        <div class="col-text">{bullets}{next_html}</div>
        <div class="col-visual"><div class="recap-grid">{grid}</div></div>
      </div>
    </div>"""


# ==============================
# LAYOUT DISPATCH
# ==============================

LAYOUT_MAP = {
    "title_hero":            layout_title_hero,
    "definition_spotlight":  layout_definition_spotlight,
    "formula_showcase":      layout_formula_showcase,
    "step_walkthrough":      layout_step_walkthrough,
    "problem_setup":         layout_problem_setup,
    "visual_explain":        layout_visual_explain,
    "visual_full":           layout_visual_full,
    "bullet_list":           layout_bullet_list,
    "key_takeaway":          layout_key_takeaway,
    "split_comparison":      layout_split_comparison,
    "recap_grid":            layout_recap_grid,
}

TYPE_FALLBACK = {
    "intro":                "title_hero",
    "definition":           "definition_spotlight",
    "concept_explanation":  "bullet_list",
    "formula_derivation":   "step_walkthrough",
    "worked_example":       "step_walkthrough",
    "recap":                "recap_grid",
}


def render_chunk_html(chunk, meta):
    layout_config = chunk.get("layout_config") or {}
    layout_name   = layout_config.get("layout", "")
    renderer = LAYOUT_MAP.get(layout_name)
    if not renderer:
        ctype    = chunk.get("type", "")
        fallback = TYPE_FALLBACK.get(ctype, "bullet_list")
        renderer = LAYOUT_MAP.get(fallback, layout_bullet_list)
    content_html = renderer(chunk, meta)
    header_meta  = make_header_meta(chunk, meta)
    return base_html(content_html, header_meta)


# ==============================
# RENDER SLIDE TO PNG
# ==============================

async def render_slide(page, html, out_path):
    await page.set_content(html, wait_until="networkidle")
    await page.wait_for_timeout(300)
    await page.screenshot(path=str(out_path))


# ==============================
# COLLECT ALL CHUNKS FOR A CHAPTER
# ==============================

def get_wav_duration(wav_path):
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def collect_chapter_chunks(merge_code):
    chunk_base = CHUNKS_DIR / merge_code
    audio_base = AUDIO_DIR  / merge_code
    if not chunk_base.is_dir():
        return []
    all_entries = []
    module_dirs = sorted([d for d in chunk_base.iterdir() if d.is_dir()])
    for mod_dir in module_dirs:
        module_id = mod_dir.name
        meta_path = mod_dir / "_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("errors"):
            print(f"  SKIP {module_id}: has chunk errors")
            continue
        timeline_path = audio_base / module_id / "timeline.json"
        if not timeline_path.exists():
            print(f"  SKIP {module_id}: no timeline.json (audio incomplete)")
            continue
        chunk_order = meta.get("chunk_order", [])
        if not chunk_order:
            continue
        for chunk_file in chunk_order:
            chunk_path = mod_dir / chunk_file
            if not chunk_path.exists():
                continue
            with open(chunk_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            wav_name = chunk_file.replace(".json", ".wav")
            audio_path = audio_base / module_id / wav_name
            duration = get_wav_duration(audio_path) if audio_path.exists() else 0.0
            all_entries.append({
                "chunk_data":  chunk_data,
                "meta":        meta,
                "chunk_file":  chunk_file,
                "module_id":   module_id,
                "audio_path":  audio_path,
                "duration":    duration,
            })
    return all_entries


# ==============================
# FFMPEG HELPERS
# ==============================

def run_ffmpeg(cmd, label):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFMPEG FAIL [{label}]")
        print(f"     {result.stderr[-500:]}")
        return False
    return True


def ffmpeg_segment(slide, audio, out, duration):
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-i", slide,
        "-i", audio,
        "-c:v", "libx264", "-tune", "stillimage",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-t", str(duration + 0.1),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-r", "1",
        out
    ], Path(out).name)


def ffmpeg_concat(segment_paths, out):
    concat_file = out.parent / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy", str(out)
    ], out.name)
    if ok:
        size_kb = out.stat().st_size // 1000
        print(f"     concat OK  size={size_kb}KB")
    return ok


def ffmpeg_with_intro_end(content_video, out):
    if not INTRO_VIDEO.exists() or not END_VIDEO.exists():
        print(f"  intro/end not found -- copying content as final")
        return run_ffmpeg(["ffmpeg", "-y", "-i", str(content_video), "-c", "copy", str(out)], out.name)
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(INTRO_VIDEO), "-i", str(content_video), "-i", str(END_VIDEO),
        "-filter_complex", "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1",
        "-pix_fmt", "yuv420p",
        str(out)
    ], out.name)


# ==============================
# ENCODE ALL SEGMENTS (parallel)
# ==============================

def encode_all_segments(entries, slides_dir):
    """Encode all slide+audio pairs into MP4 segments. Returns ordered list of segment paths."""
    jobs = []
    segment_order = []
    for entry in entries:
        slide_path = entry.get("slide_path")
        audio_path = entry["audio_path"]
        duration   = entry["duration"]
        if not slide_path or not Path(slide_path).exists():
            continue
        if not audio_path.exists():
            continue
        if duration <= 0:
            continue
        safe_name = entry["chunk_file"].replace(".json", "").replace(" ", "_")
        seg_out = slides_dir / f"seg_{entry['module_id']}_{safe_name}.mp4"
        segment_order.append((seg_out, entry["module_id"]))
        if not seg_out.exists():
            jobs.append((str(slide_path), str(audio_path), str(seg_out), duration))

    cached = len(segment_order) - len(jobs)
    if cached:
        log_progress(f"  {cached} segments cached, {len(jobs)} to encode")

    if jobs:
        log_progress(f"  Encoding {len(jobs)} segments ({FFMPEG_WORKERS} workers)...")
        failed = 0
        with ThreadPoolExecutor(max_workers=FFMPEG_WORKERS) as pool:
            futures = {pool.submit(ffmpeg_segment, s, a, o, d): o for s, a, o, d in jobs}
            done = 0
            for future in as_completed(futures):
                done += 1
                if not future.result():
                    failed += 1
                if done % 20 == 0 or done == len(jobs):
                    log_progress(f"  Encoded {done}/{len(jobs)} segments")
        if failed:
            log_progress(f"  WARNING: {failed} segments failed encoding")

    return [(p, mid) for p, mid in segment_order if p.exists()]


# ==============================
# ASSEMBLE CONCEPT + CHAPTER VIDEOS
# ==============================

def assemble_concept_videos(segments_with_module, slides_dir, merge_code):
    """Assemble per-concept videos with intro+end. Returns list of concept video info dicts."""
    # Group segments by module_id
    from collections import OrderedDict
    module_segments = OrderedDict()
    for seg_path, module_id in segments_with_module:
        if module_id not in module_segments:
            module_segments[module_id] = []
        module_segments[module_id].append(seg_path)

    concept_videos = []
    for module_id, segs in module_segments.items():
        if not segs:
            continue

        concept_content = slides_dir / f"concept_content_{module_id}.mp4"
        concept_final   = slides_dir / f"{module_id}.mp4"

        # Concat segments for this concept
        ok = ffmpeg_concat(segs, concept_content)
        if not ok or not concept_content.exists():
            log_progress(f"  WARNING: concat failed for {module_id}")
            continue

        # Add intro + end
        ffmpeg_with_intro_end(concept_content, concept_final)
        concept_content.unlink(missing_ok=True)  # cleanup intermediate

        if concept_final.exists():
            concept_videos.append({
                "module_id": module_id,
                "path": concept_final,
                "segment_count": len(segs),
            })

    return concept_videos


def assemble_full_chapter_video(concept_videos, slides_dir, output_path):
    """Concat all concept videos into one full chapter video with intro+end."""
    if not concept_videos:
        return False

    # Concat all concept content (without their individual intro/end, use segments directly)
    # Actually, for full chapter we want: intro + all segments + end (not nested intros)
    # So we need the raw segments. But we already have concept videos with intro/end.
    # Better approach: concat all concept videos as-is (each already has intro/end per concept)
    # OR: build full chapter from raw segments with single intro/end
    # User wants both to have intro+end, so full chapter = intro + all_raw_segments + end

    return True  # handled in main flow


# ==============================
# ASSEMBLE CHAPTER VIDEO (from raw segments)
# ==============================

def assemble_chapter_from_segments(segments_with_module, slides_dir, output_path):
    """Build full chapter video: intro + all segments concatenated + end."""
    all_segs = [p for p, _ in segments_with_module]
    if not all_segs:
        log_progress("  Zero segments -- nothing to assemble.")
        return False

    content_video = slides_dir / "chapter_content.mp4"
    log_progress(f"  Concatenating {len(all_segs)} segments for full chapter...")
    ok = ffmpeg_concat(all_segs, content_video)
    if not ok or not content_video.exists():
        log_progress("  Concat failed")
        return False

    log_progress(f"  Adding intro/end to full chapter...")
    ffmpeg_with_intro_end(content_video, output_path)
    content_video.unlink(missing_ok=True)

    if output_path.exists():
        size_mb = output_path.stat().st_size // 1_000_000
        log_progress(f"  Full chapter: {output_path.name}  ({size_mb} MB)")
        return True
    return False


# ==============================
# POST-PROCESSING: Upload + Save + Cleanup
# ==============================

def get_video_duration_seconds(video_path):
    """Get video duration via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def post_process_chapter(merge_code, full_video_path, concept_videos, slides_dir, total_slides):
    """Upload concept videos + full chapter to S3, save to MongoDB, delete local.
    Returns True only if ALL steps succeed."""

    ch_meta = CHAPTER_META.get(merge_code, {})
    modules_info = collect_module_info(merge_code)
    now = datetime.now(timezone.utc).isoformat()

    # ── STEP 1: Upload concept videos to S3 ──
    module_docs = []
    for cv in concept_videos:
        module_id = cv["module_id"]
        cv_path   = cv["path"]
        cv_size   = cv_path.stat().st_size / 1_000_000
        cv_dur    = get_video_duration_seconds(cv_path)
        s3_key    = f"videos/{merge_code}/{module_id}.mp4"

        log_progress(f"  {merge_code}: Uploading {module_id} ({cv_size:.1f} MB)...")
        try:
            s3_url = retry_operation(upload_to_s3, merge_code, f"S3_CONCEPT_{module_id}", cv_path, s3_key)
        except Exception:
            log_error(merge_code, f"S3_CONCEPT_{module_id}_FINAL", f"Failed after {MAX_RETRIES} attempts")
            return False

        # Find module title from modules_info
        mod_info = next((m for m in modules_info if m["module_id"] == module_id), {})

        module_docs.append({
            "module_id": module_id,
            "module_title": mod_info.get("module_title", ""),
            "total_slides": mod_info.get("total_slides", cv["segment_count"]),
            "duration_seconds": round(cv_dur, 2),
            "duration_display": f"{int(cv_dur // 60)}m {int(cv_dur % 60)}s",
            "file_size_mb": round(cv_size, 2),
            "s3_url": s3_url,
            "s3_key": s3_key,
        })

    log_progress(f"  {merge_code}: All {len(concept_videos)} concept videos uploaded")

    # ── STEP 2: Upload full chapter video to S3 ──
    full_size = full_video_path.stat().st_size / 1_000_000
    full_dur  = get_video_duration_seconds(full_video_path)
    full_s3_key = f"videos/{merge_code}/{merge_code}_full.mp4"

    log_progress(f"  {merge_code}: Uploading full chapter ({full_size:.1f} MB)...")
    try:
        full_s3_url = retry_operation(upload_to_s3, merge_code, "S3_FULL", full_video_path, full_s3_key)
        log_progress(f"  {merge_code}: Full chapter S3 OK")
    except Exception:
        log_error(merge_code, "S3_FULL_FINAL", f"Failed after {MAX_RETRIES} attempts")
        return False

    # ── STEP 3: Save to MongoDB ──
    doc = {
        "merge_code": merge_code,
        "chapter_name": ch_meta.get("chapter_name", ""),
        "domain": ch_meta.get("domain", ""),
        "classes": ch_meta.get("classes", []),
        "chapter_numbers": ch_meta.get("chapters", []),
        "modules": module_docs,
        "total_slides": total_slides,
        "total_concepts": len(module_docs),
        "full_video": {
            "duration_seconds": round(full_dur, 2),
            "duration_display": f"{int(full_dur // 60)}m {int(full_dur % 60)}s",
            "file_size_mb": round(full_size, 2),
            "s3_url": full_s3_url,
            "s3_key": full_s3_key,
        },
        "s3_bucket": S3_BUCKET,
        "status": "published",
        "generated_at": now,
    }

    log_progress(f"  {merge_code}: Saving to MongoDB...")
    try:
        retry_operation(save_to_mongodb, merge_code, "MONGODB_SAVE", doc)
        log_progress(f"  {merge_code}: MongoDB OK")
    except Exception:
        log_error(merge_code, "MONGODB_SAVE_FINAL", f"Failed after {MAX_RETRIES} attempts")
        return False

    # ── STEP 4: Delete local files ──
    log_progress(f"  {merge_code}: Cleaning up local files...")
    try:
        full_video_path.unlink(missing_ok=True)
        for cv in concept_videos:
            cv["path"].unlink(missing_ok=True)
        if slides_dir.exists():
            shutil.rmtree(slides_dir)
        log_progress(f"  {merge_code}: Cleanup OK")
    except Exception as e:
        log_progress(f"  {merge_code}: Cleanup warning (non-fatal): {e}")

    return True


# ==============================
# COLLECT JOBS
# ==============================

def collect_jobs():
    if not CHUNKS_DIR.is_dir():
        return []
    return sorted([
        d.name for d in CHUNKS_DIR.iterdir()
        if d.is_dir() and not d.name.endswith(".json")
    ])


# ==============================
# MAIN
# ==============================

async def main():
    log_progress("\n========================================")
    log_progress("Video Pipeline v3 -- Render + S3 + MongoDB")
    log_progress("========================================\n")

    # Validate config
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        log_progress("FATAL: AWS credentials not set. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        sys.exit(1)

    merge_codes = collect_jobs()
    if not merge_codes:
        log_progress(f"No chapters found in {CHUNKS_DIR}")
        return

    if DEBUG:
        merge_codes = merge_codes[:1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_progress(f"Total chapters found: {len(merge_codes)}")

    # Check MongoDB for already-published chapters (skip on rerun)
    completed_codes = set()
    try:
        col = get_mongo_collection()
        completed_codes = set(
            doc["merge_code"] for doc in col.find({"status": "published"}, {"merge_code": 1})
        )
        if completed_codes:
            log_progress(f"Already published in MongoDB: {len(completed_codes)} — will skip")
    except Exception as e:
        log_progress(f"WARNING: Could not check MongoDB: {e}")
        log_progress("Continuing — will attempt all chapters")

    remaining = [mc for mc in merge_codes if mc not in completed_codes]
    log_progress(f"Chapters to process: {len(remaining)}\n")

    if not remaining:
        log_progress("All chapters already published. Nothing to do.")
        return

    succeeded = 0
    failed = 0
    pipeline_start = time.time()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page    = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for idx, merge_code in enumerate(remaining, 1):
            chapter_start = time.time()
            full_video_path = OUTPUT_DIR / f"{merge_code}_full.mp4"
            slides_dir      = OUTPUT_DIR / f"slides_{merge_code}"

            log_progress(f"\n[{idx}/{len(remaining)}] ── {merge_code} ──")

            try:
                # ── STEP 1: Collect chunks ──
                entries = collect_chapter_chunks(merge_code)
                if not entries:
                    log_error(merge_code, "COLLECT", "No chunks/audio found")
                    log_progress(f"  {merge_code}: SKIPPED (no data)")
                    continue

                total_slides = len(entries)
                modules_in_chapter = list(dict.fromkeys(e["module_id"] for e in entries))
                log_progress(f"  {merge_code}: {total_slides} slides, {len(modules_in_chapter)} concepts")

                slides_dir.mkdir(parents=True, exist_ok=True)

                # ── STEP 2: Render slides to PNG ──
                for si, entry in enumerate(entries, 1):
                    safe_name = entry["chunk_file"].replace(".json", "")
                    slide_name = f"{entry['module_id']}_{safe_name}.png"
                    slide_path = slides_dir / slide_name
                    entry["slide_path"] = slide_path

                    if slide_path.exists():
                        continue

                    html = render_chunk_html(entry["chunk_data"], entry["meta"])
                    if si % 20 == 1 or si == total_slides:
                        log_progress(f"  {merge_code}: Rendering slide {si}/{total_slides}")
                    await render_slide(page, html, slide_path)

                # ── STEP 3: Encode segments (parallel) ──
                log_progress(f"  {merge_code}: Encoding segments...")
                segments_with_module = encode_all_segments(entries, slides_dir)

                if not segments_with_module:
                    log_error(merge_code, "ENCODE", "Zero segments encoded")
                    log_progress(f"  {merge_code}: FAILED at encoding. STOPPING PIPELINE.")
                    failed += 1
                    break

                log_progress(f"  {merge_code}: {len(segments_with_module)} segments ready")

                # ── STEP 4: Assemble concept videos (per module, each with intro+end) ──
                log_progress(f"  {merge_code}: Building {len(modules_in_chapter)} concept videos...")
                concept_videos = assemble_concept_videos(segments_with_module, slides_dir, merge_code)

                if not concept_videos:
                    log_error(merge_code, "CONCEPTS", "No concept videos assembled")
                    failed += 1
                    break

                log_progress(f"  {merge_code}: {len(concept_videos)} concept videos built")

                # ── STEP 5: Assemble full chapter video (intro + all segments + end) ──
                log_progress(f"  {merge_code}: Building full chapter video...")
                ok = assemble_chapter_from_segments(segments_with_module, slides_dir, full_video_path)

                if not ok or not full_video_path.exists():
                    log_error(merge_code, "FULL_CHAPTER", "Full chapter assembly failed")
                    failed += 1
                    break

                # ── STEP 6: Upload all to S3 + Save to MongoDB + Delete local ──
                log_progress(f"  {merge_code}: Uploading {len(concept_videos)} concepts + 1 full chapter...")
                ok = post_process_chapter(merge_code, full_video_path, concept_videos, slides_dir, total_slides)

                if not ok:
                    log_progress(f"  {merge_code}: FAILED at post-processing. STOPPING PIPELINE.")
                    failed += 1
                    break

                # ── SUCCESS ──
                elapsed = time.time() - chapter_start
                succeeded += 1
                log_progress(f"  {merge_code}: COMPLETE in {int(elapsed)}s ({succeeded}/{len(remaining)} done)")

            except Exception as e:
                log_error(merge_code, "UNEXPECTED", f"{type(e).__name__}: {e}")
                log_progress(f"  {merge_code}: UNEXPECTED ERROR. STOPPING PIPELINE.")
                traceback.print_exc()
                failed += 1
                break

        await browser.close()

    # ── SUMMARY ──
    total_time = time.time() - pipeline_start
    hours = int(total_time // 3600)
    mins  = int((total_time % 3600) // 60)

    log_progress(f"\n========================================")
    log_progress(f"PIPELINE COMPLETE")
    log_progress(f"  Succeeded: {succeeded}")
    log_progress(f"  Failed:    {failed}")
    log_progress(f"  Skipped:   {len(completed_codes)}")
    log_progress(f"  Total time: {hours}h {mins}m")
    log_progress(f"  Error log:  {ERROR_LOG}")
    log_progress(f"  Progress:   {PROGRESS_LOG}")
    log_progress(f"========================================")


if __name__ == "__main__":
    asyncio.run(main())
