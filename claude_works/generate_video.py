"""
Video generation pipeline for educational math content.
- Reads chunks from claude_works/chunks/ and audio from claude_works/audio/
- Renders HTML slides via Playwright → PNG screenshots
- Assembles video with FFmpeg (slide + audio → segment → concat → intro/outro)
- 8 color themes, randomly assigned per concept
- Produces concept-wise AND module-wise merged videos
- Polls for new ready modules
"""
import json
import hashlib
import asyncio
import subprocess
import time
import re
from pathlib import Path
from playwright.async_api import async_playwright

# ==========================================
# CONFIG
# ==========================================
CHUNKS_FOLDER = Path("claude_works/chunks")
AUDIO_FOLDER  = Path("claude_works/audio")
VIDEOS_FOLDER = Path("claude_works/videos")
INTRO_VIDEO   = Path("intro_v0.mp4")
END_VIDEO     = Path("end_v0.mp4")

WIDTH, HEIGHT = 1920, 1080
FPS           = 30
DEBUG         = False
POLL_INTERVAL = 30
MAX_EMPTY_POLLS = 30

VIDEOS_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================
# 8 COLOR THEMES
# ==========================================
THEMES = [
    {
        "name": "Cyan",
        "brand": "#00c6ff",
        "bg": "#020202",
        "bg_gradient_focus": "#001a25",
        "text": "#ffffff",
        "subtext": "#cfcfcf",
        "accent_rgb": "0,198,255",
        "exam_tip_accent": "#ffc800",
        "exam_tip_rgb": "255,200,0",
    },
    {
        "name": "Violet",
        "brand": "#b366ff",
        "bg": "#050010",
        "bg_gradient_focus": "#1a0033",
        "text": "#ffffff",
        "subtext": "#d4c6e8",
        "accent_rgb": "179,102,255",
        "exam_tip_accent": "#ff9d00",
        "exam_tip_rgb": "255,157,0",
    },
    {
        "name": "Emerald",
        "brand": "#00e676",
        "bg": "#010d06",
        "bg_gradient_focus": "#001a0d",
        "text": "#ffffff",
        "subtext": "#c8e6d0",
        "accent_rgb": "0,230,118",
        "exam_tip_accent": "#ffab40",
        "exam_tip_rgb": "255,171,64",
    },
    {
        "name": "Amber",
        "brand": "#ffab00",
        "bg": "#0d0800",
        "bg_gradient_focus": "#1a1000",
        "text": "#ffffff",
        "subtext": "#e8dcc8",
        "accent_rgb": "255,171,0",
        "exam_tip_accent": "#00e5ff",
        "exam_tip_rgb": "0,229,255",
    },
    {
        "name": "Rose",
        "brand": "#ff4081",
        "bg": "#0d0005",
        "bg_gradient_focus": "#1a000d",
        "text": "#ffffff",
        "subtext": "#e8c8d4",
        "accent_rgb": "255,64,129",
        "exam_tip_accent": "#69f0ae",
        "exam_tip_rgb": "105,240,174",
    },
    {
        "name": "Indigo",
        "brand": "#536dfe",
        "bg": "#000510",
        "bg_gradient_focus": "#000d24",
        "text": "#ffffff",
        "subtext": "#c8d0e8",
        "accent_rgb": "83,109,254",
        "exam_tip_accent": "#ffd740",
        "exam_tip_rgb": "255,215,64",
    },
    {
        "name": "Teal",
        "brand": "#1de9b6",
        "bg": "#000d0a",
        "bg_gradient_focus": "#001a14",
        "text": "#ffffff",
        "subtext": "#c8e8e0",
        "accent_rgb": "29,233,182",
        "exam_tip_accent": "#ff6e40",
        "exam_tip_rgb": "255,110,64",
    },
    {
        "name": "Gold",
        "brand": "#ffd600",
        "bg": "#0d0b00",
        "bg_gradient_focus": "#1a1500",
        "text": "#ffffff",
        "subtext": "#e8e4c8",
        "accent_rgb": "255,214,0",
        "exam_tip_accent": "#448aff",
        "exam_tip_rgb": "68,138,255",
    },
]


def pick_theme(module_id: str) -> dict:
    h = int(hashlib.md5(module_id.encode()).hexdigest(), 16)
    return THEMES[h % len(THEMES)]


# ==========================================
# CHUNK NORMALIZATION
# ==========================================
def infer_type_from_filename(filename: str) -> str:
    """Extract type from chunk filename like '002_definition_0.json'."""
    stem = filename.replace(".json", "")
    parts = stem.split("_")
    if len(parts) < 2:
        return "generic"
    raw_type = parts[1]
    mapping = {
        "intro": "intro",
        "definition": "definition",
        "property": "concept_explanation",
        "formula": "concept_explanation",
        "example": "worked_example",
        "theorem": "theorem",
        "recap": "recap",
        "key": "key_takeaway",
    }
    return mapping.get(raw_type, "concept_explanation")


def normalize_chunk(raw: dict, filename: str = "") -> dict:
    """Normalize various chunk schema variants into canonical form."""
    content = raw.get("content", {}) if isinstance(raw.get("content"), dict) else {}

    chunk_type = (
        raw.get("type")
        or raw.get("slide_type")
        or raw.get("chunk_type")
        or infer_type_from_filename(filename)
        or "generic"
    )

    # Map variant type names
    type_map = {
        "formula_derivation": "concept_explanation",
        "step_walkthrough": "step_walkthrough",
        "worked_example": "worked_example",
        "key_takeaway": "key_takeaway",
    }
    chunk_type = type_map.get(chunk_type, chunk_type)

    return {
        "chunk_id": raw.get("chunk_id") or raw.get("slide_id", ""),
        "type": chunk_type,
        "slide_title": raw.get("slide_title") or raw.get("display_title", ""),
        "display_bullets": raw.get("display_bullets") or content.get("display_bullets") or [],
        "script": raw.get("script") or content.get("script", ""),
        "script_display": raw.get("script_display") or content.get("script_display", ""),
        "visual": raw.get("visual") or raw.get("visual_aid") or {"type": "none"},
        "exam_tip": raw.get("exam_tip"),
        "coverage_summary": raw.get("coverage_summary") or content.get("coverage_summary"),
        "next_modules": raw.get("next_modules") or content.get("next_modules"),
        "prerequisites_display": raw.get("prerequisites_display") or content.get("prerequisites_display"),
    }


# ==========================================
# BASE HTML WRAPPER (themed)
# ==========================================
def base_html(body: str, theme: dict, meta: dict, chunk: dict) -> str:
    T = theme
    sidebar_info = f"""
    <div>{meta.get('class','')}ನೇ ತರಗತಿ</div>
    <div>ಅಧ್ಯಾಯ {meta.get('chapter','')}</div>
    <div style='margin-top:8px;font-size:14px;color:rgba(255,255,255,0.2)'>{chunk.get('chunk_id','')}</div>"""

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
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {WIDTH}px; height: {HEIGHT}px;
    background: {T['bg']};
    background-image: radial-gradient(circle at 75% 50%, {T['bg_gradient_focus']} 0%, {T['bg']} 65%);
    font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
    color: {T['text']};
    display: flex;
    overflow: hidden;
  }}
  .sidebar {{
    width: 420px; height: 100%;
    display: flex; flex-direction: column;
    align-items: flex-start;
    padding: 80px 0 80px 70px;
    border-right: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
  }}
  .brand {{
    font-size: 20px; font-weight: 600;
    letter-spacing: 4px; color: {T['brand']};
    text-transform: uppercase;
    border-left: 3px solid {T['brand']};
    padding-left: 18px; line-height: 1;
  }}
  .sidebar-meta {{
    margin-top: auto;
    font-size: 18px; color: rgba(255,255,255,0.3);
    line-height: 1.8;
  }}
  .content {{
    flex-grow: 1;
    padding: 70px 80px 70px 70px;
    display: flex; flex-direction: column;
    justify-content: center; overflow: hidden;
  }}
  .slide-title {{
    font-size: 52px; font-weight: 800;
    line-height: 1.2; margin-bottom: 18px;
    color: {T['text']};
  }}
  .accent-line {{
    width: 100px; height: 4px;
    background: {T['brand']};
    margin-bottom: 44px; border-radius: 2px;
    box-shadow: 0 0 16px rgba({T['accent_rgb']},0.45);
  }}
  .two-col {{
    display: flex; gap: 60px;
    align-items: flex-start; flex-grow: 1;
  }}
  .col-text {{ flex: 1; }}
  .col-visual {{ flex: 0 0 520px; }}
  .bullet-list {{ list-style: none; }}
  .bullet-list li {{
    font-size: 34px; line-height: 1.55;
    margin-bottom: 24px; display: flex;
    align-items: flex-start; color: {T['subtext']};
  }}
  .bullet-node {{
    width: 7px; height: 26px;
    background: {T['brand']}; margin-top: 14px;
    margin-right: 24px; flex-shrink: 0;
    border-radius: 1px;
  }}
  /* STEP CARDS */
  .step-card {{
    border-radius: 10px; padding: 16px 22px;
    margin-bottom: 14px;
    border-left: 3px solid rgba({T['accent_rgb']},0.3);
    font-size: 30px; line-height: 1.5;
    color: {T['subtext']};
    background: rgba({T['accent_rgb']},0.06);
  }}
  .step-num {{
    font-size: 20px; font-weight: 600;
    color: {T['brand']}; margin-bottom: 4px;
  }}
  .justification {{
    font-size: 22px; color: rgba(255,255,255,0.45);
    margin-top: 4px; font-style: italic;
  }}
  /* DEFINITION CARD */
  .def-card {{
    background: rgba({T['accent_rgb']},0.07);
    border: 1px solid rgba({T['accent_rgb']},0.2);
    border-radius: 12px; padding: 32px 36px;
    margin-bottom: 24px;
  }}
  .def-term {{
    font-size: 34px; font-weight: 700;
    color: {T['brand']}; margin-bottom: 12px;
  }}
  .def-body {{ font-size: 34px; line-height: 1.6; color: {T['subtext']}; }}
  /* FORMULA BOX */
  .formula-box {{
    background: rgba({T['accent_rgb']},0.06);
    border: 1px solid rgba({T['accent_rgb']},0.25);
    border-radius: 12px; padding: 28px 36px;
    text-align: center; font-size: 42px;
  }}
  .formula-label {{
    font-size: 22px; color: rgba(255,255,255,0.4);
    margin-bottom: 14px; text-transform: uppercase;
    letter-spacing: 2px;
  }}
  /* TABLE */
  table {{
    border-collapse: collapse; width: 100%;
    font-size: 28px;
  }}
  th {{
    background: rgba({T['accent_rgb']},0.15);
    color: {T['brand']}; padding: 14px 20px;
    text-align: left; font-weight: 600;
    border-bottom: 1px solid rgba({T['accent_rgb']},0.3);
  }}
  td {{
    padding: 12px 20px; color: {T['subtext']};
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}
  tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
  /* EXAM TIP */
  .exam-tip {{
    background: rgba({T['exam_tip_rgb']},0.07);
    border: 1px solid rgba({T['exam_tip_rgb']},0.25);
    border-radius: 10px; padding: 20px 26px;
    font-size: 26px; margin-top: 24px;
  }}
  .exam-tip-label {{
    font-size: 18px; font-weight: 700;
    color: {T['exam_tip_accent']}; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px;
  }}
  /* RECAP GRID */
  .recap-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px; margin-top: 8px;
  }}
  .recap-cell {{
    background: rgba({T['accent_rgb']},0.07);
    border: 1px solid rgba({T['accent_rgb']},0.15);
    border-radius: 10px; padding: 20px;
    text-align: center;
  }}
  .recap-count {{
    font-size: 48px; font-weight: 800;
    color: {T['brand']}; line-height: 1;
  }}
  .recap-label {{
    font-size: 20px; color: {T['subtext']};
    margin-top: 8px;
  }}
  /* THEOREM CARD */
  .theorem-card {{
    background: rgba({T['accent_rgb']},0.08);
    border-left: 4px solid {T['brand']};
    border-radius: 0 12px 12px 0;
    padding: 32px 36px;
    margin-bottom: 24px;
  }}
  .theorem-label {{
    font-size: 20px; font-weight: 700;
    color: {T['brand']}; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 12px;
  }}
  /* KEY TAKEAWAY */
  .takeaway-card {{
    background: rgba({T['exam_tip_rgb']},0.05);
    border: 1px solid rgba({T['exam_tip_rgb']},0.2);
    border-radius: 12px; padding: 24px 30px;
    margin-bottom: 16px;
  }}
  /* DIAGRAM PLACEHOLDER */
  .diagram-placeholder {{
    border: 2px dashed rgba({T['accent_rgb']},0.3);
    border-radius: 12px; padding: 32px;
    text-align: center;
  }}
  .diagram-icon {{
    font-size: 48px; margin-bottom: 16px;
    opacity: 0.6;
  }}
  .diagram-desc {{
    font-size: 24px; color: {T['subtext']};
    line-height: 1.5; margin-bottom: 12px;
  }}
  .diagram-sig {{
    font-size: 20px; color: rgba(255,255,255,0.35);
    font-style: italic;
  }}
</style>
</head>
<body>
  <div class="sidebar">
    <div class="brand">SRINIVAS IAS<br>ACADEMY</div>
    <div class="sidebar-meta">{sidebar_info}</div>
  </div>
  {body}
</body>
</html>"""


# ==========================================
# VISUAL HTML BUILDER
# ==========================================
def build_visual_html(visual: dict, theme: dict) -> str:
    if not visual or visual.get("type") == "none":
        return ""

    vtype = visual.get("type", "")
    T = theme

    if vtype == "formula_box":
        latex = visual.get("latex", "")
        if not latex:
            return ""
        return f'<div class="formula-box">$${latex}$$</div>'

    if vtype == "table":
        headers = visual.get("headers")
        rows = visual.get("rows")
        if not headers and visual.get("latex"):
            try:
                parsed = json.loads(visual["latex"])
                headers = parsed.get("headers", [])
                rows = parsed.get("rows", [])
            except Exception:
                pass
        if headers and rows:
            th = "".join(f"<th>{h}</th>" for h in headers)
            tr = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                for row in rows
            )
            caption = visual.get("caption") or ""
            cap_html = f'<div style="font-size:20px;color:rgba(255,255,255,0.4);margin-bottom:10px">{caption}</div>' if caption else ""
            return f"{cap_html}<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"
        # Fallback: show placeholder_note
        note = visual.get("placeholder_note", "")
        if note:
            return f'<div style="font-size:24px;color:rgba(255,255,255,0.4);padding:20px">{note}</div>'
        return ""

    if vtype == "diagram":
        desc = visual.get("description", "")
        labels = visual.get("labels", "")
        sig = visual.get("mathematical_significance", "")
        labels_html = f'<div style="font-size:20px;color:{T["brand"]};margin-bottom:10px">{labels}</div>' if labels else ""
        sig_html = f'<div class="diagram-sig">{sig}</div>' if sig else ""
        return f"""
        <div class="diagram-placeholder">
          <div class="diagram-icon">📐</div>
          {labels_html}
          <div class="diagram-desc">{desc}</div>
          {sig_html}
        </div>"""

    return ""


def build_exam_tip(chunk: dict, theme: dict) -> str:
    et = chunk.get("exam_tip")
    if not et:
        return ""
    pattern = et.get("question_pattern", "")
    skill = et.get("skill_tested", "")
    distractor = et.get("distractor", "")
    return f"""
    <div class="exam-tip">
      <div class="exam-tip-label">ಪರೀಕ್ಷಾ ಸೂಚನೆ · {pattern}</div>
      <div style="color:{theme['subtext']}">{skill}</div>
      <div style="margin-top:6px;color:rgba({theme['exam_tip_rgb']},0.6);font-size:22px">
        {distractor}
      </div>
    </div>"""


# ==========================================
# HTML TEMPLATES PER CHUNK TYPE
# ==========================================
def _bullets_html(bullets: list) -> str:
    if not bullets:
        return ""
    items = "".join(f"<li><span class='bullet-node'></span><span>{b}</span></li>" for b in bullets)
    return f'<ul class="bullet-list">{items}</ul>'


def html_intro(chunk: dict, meta: dict, theme: dict) -> str:
    T = theme
    prereqs = ""
    if chunk.get("prerequisites_display"):
        items = "".join(f"<li><span class='bullet-node'></span>{p}</li>"
                        for p in chunk["prerequisites_display"])
        prereqs = f"<div style='font-size:24px;color:rgba(255,255,255,0.4);margin-bottom:12px;text-transform:uppercase;letter-spacing:2px'>ಪೂರ್ವಾಪೇಕ್ಷಿತಗಳು</div><ul class='bullet-list'>{items}</ul>"

    bullets = _bullets_html(chunk.get("display_bullets", []))

    body = f"""
  <div class="content" style="justify-content:center">
    <div style="font-size:24px;color:{T['brand']};letter-spacing:3px;text-transform:uppercase;margin-bottom:20px">
      {meta.get('domain','')} · {meta.get('class','')}ನೇ ತರಗತಿ
    </div>
    <div class="slide-title" style="font-size:64px">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    {bullets}
    <div style="margin-top:36px">{prereqs}</div>
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_definition(chunk: dict, meta: dict, theme: dict, visual_html: str = "") -> str:
    bullets = _bullets_html(chunk.get("display_bullets", []))

    if visual_html:
        inner = f"""
        <div class="two-col">
          <div class="col-text">{bullets}</div>
          <div class="col-visual">{visual_html}</div>
        </div>"""
    else:
        inner = bullets

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    {inner}
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_concept_explanation(chunk: dict, meta: dict, theme: dict, visual_html: str = "") -> str:
    bullets = _bullets_html(chunk.get("display_bullets", []))
    exam_tip = build_exam_tip(chunk, theme)

    if visual_html:
        inner = f"""
        <div class="two-col">
          <div class="col-text">{bullets}</div>
          <div class="col-visual">{visual_html}</div>
        </div>"""
    else:
        inner = bullets

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    {inner}
    {exam_tip}
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_step_walkthrough(chunk: dict, meta: dict, theme: dict) -> str:
    bullets = chunk.get("display_bullets", [])
    cards = ""
    for i, b in enumerate(bullets):
        cards += f"""
        <div class="step-card">
          <div class="step-num">ಹಂತ {i+1}</div>
          <div>{b}</div>
        </div>"""

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    <div style="overflow-y:auto;max-height:780px">{cards}</div>
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_worked_example(chunk: dict, meta: dict, theme: dict, visual_html: str = "") -> str:
    T = theme
    bullets = chunk.get("display_bullets", [])
    exam_tip = build_exam_tip(chunk, theme)

    # First bullet is often the problem statement
    problem_html = ""
    step_bullets = bullets
    if bullets:
        problem_html = f"""
        <div style="font-size:30px;color:{T['subtext']};margin-bottom:24px;
                    background:rgba(255,255,255,0.04);padding:18px 24px;
                    border-radius:8px;border-left:3px solid rgba({T['accent_rgb']},0.3)">
          {bullets[0]}
        </div>"""
        step_bullets = bullets[1:]

    steps_html = ""
    for i, b in enumerate(step_bullets):
        steps_html += f"""
        <div class="step-card">
          <div class="step-num">ಹಂತ {i+1}</div>
          <div>{b}</div>
        </div>"""

    if visual_html:
        inner = f"""
        <div class="two-col">
          <div class="col-text">
            {problem_html}
            <div style="overflow-y:auto;max-height:560px">{steps_html}</div>
          </div>
          <div class="col-visual">{visual_html}</div>
        </div>"""
    else:
        inner = f"""
        {problem_html}
        <div style="overflow-y:auto;max-height:640px">{steps_html}</div>"""

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    {inner}
    {exam_tip}
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_key_takeaway(chunk: dict, meta: dict, theme: dict) -> str:
    bullets = chunk.get("display_bullets", [])
    exam_tip = build_exam_tip(chunk, theme)

    items = ""
    for b in bullets:
        items += f"""
        <div class="takeaway-card">
          <div style="font-size:32px;line-height:1.5;color:{theme['subtext']}">{b}</div>
        </div>"""

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    <div style="overflow-y:auto;max-height:700px">{items}</div>
    {exam_tip}
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_theorem(chunk: dict, meta: dict, theme: dict) -> str:
    T = theme
    bullets = _bullets_html(chunk.get("display_bullets", []))

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    <div class="theorem-card">
      <div class="theorem-label">ಪ್ರಮೇಯ</div>
      {bullets}
    </div>
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_recap(chunk: dict, meta: dict, theme: dict) -> str:
    T = theme
    bullets = _bullets_html(chunk.get("display_bullets", []))

    cs = chunk.get("coverage_summary") or {}
    # Handle both naming conventions
    cells = [
        ("ವ್ಯಾಖ್ಯೆಗಳು",     cs.get("definitions_covered", cs.get("definitions", 0))),
        ("ಸೂತ್ರಗಳು",        cs.get("formulas_covered", cs.get("formulas", 0))),
        ("ಪ್ರಮೇಯಗಳು",       cs.get("theorems_covered", cs.get("theorems", 0))),
        ("ಗುಣಧರ್ಮಗಳು",      cs.get("properties_covered", cs.get("properties", 0))),
        ("ಉದಾಹರಣೆಗಳು",      cs.get("worked_examples_covered", cs.get("examples", 0))),
    ]
    grid = "".join(
        f'<div class="recap-cell"><div class="recap-count">{v}</div>'
        f'<div class="recap-label">{k}</div></div>'
        for k, v in cells
    )

    next_mods = chunk.get("next_modules") or []
    next_html = ""
    if next_mods:
        if isinstance(next_mods[0], dict):
            items = " · ".join(m.get("title", "") for m in next_mods)
        else:
            items = " · ".join(str(m) for m in next_mods)
        next_html = f"""
        <div style="margin-top:24px;font-size:22px;color:rgba(255,255,255,0.35)">
          ಮುಂದೆ: <span style="color:{T['brand']}">{items}</span>
        </div>"""

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title', 'ಪುನರಾವರ್ತನೆ')}</div>
    <div class="accent-line"></div>
    <div class="two-col">
      <div class="col-text">
        {bullets}
        {next_html}
      </div>
      <div class="col-visual">
        <div class="recap-grid">{grid}</div>
      </div>
    </div>
  </div>"""
    return base_html(body, theme, meta, chunk)


def html_generic(chunk: dict, meta: dict, theme: dict, visual_html: str = "") -> str:
    """Fallback for unknown chunk types — same as definition layout."""
    return html_definition(chunk, meta, theme, visual_html)


# ==========================================
# TEMPLATE DISPATCHER
# ==========================================
def render_template(chunk: dict, meta: dict, theme: dict, visual_html: str) -> str:
    ctype = chunk.get("type", "generic")
    dispatch = {
        "intro": lambda: html_intro(chunk, meta, theme),
        "definition": lambda: html_definition(chunk, meta, theme, visual_html),
        "concept_explanation": lambda: html_concept_explanation(chunk, meta, theme, visual_html),
        "step_walkthrough": lambda: html_step_walkthrough(chunk, meta, theme),
        "worked_example": lambda: html_worked_example(chunk, meta, theme, visual_html),
        "key_takeaway": lambda: html_key_takeaway(chunk, meta, theme),
        "theorem": lambda: html_theorem(chunk, meta, theme),
        "recap": lambda: html_recap(chunk, meta, theme),
    }
    fn = dispatch.get(ctype, lambda: html_generic(chunk, meta, theme, visual_html))
    return fn()


# ==========================================
# SLIDE PLAN BUILDER
# ==========================================
def build_slide_plan(meta: dict, chunk_dir: Path, audio_dir: Path,
                     theme: dict, slides_dir: Path) -> list[dict]:
    """
    Returns list of: {chunk_file, wav_path, slide_path, html, duration}
    One entry per timeline segment (1 chunk = 1 slide).
    """
    timeline_path = audio_dir / "timeline.json"
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    segments = timeline.get("segments", [])
    plan = []

    for seg in segments:
        chunk_file = seg["chunk_file"]
        wav_file = seg["wav_file"]
        chunk_path = chunk_dir / chunk_file
        wav_path = audio_dir / wav_file

        if not chunk_path.exists():
            print(f"    SKIP: chunk missing {chunk_file}")
            continue
        if not wav_path.exists():
            print(f"    SKIP: wav missing {wav_file}")
            continue

        with open(chunk_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        chunk = normalize_chunk(raw, chunk_file)
        visual = chunk["visual"]
        visual_html = build_visual_html(visual, theme)
        html = render_template(chunk, meta, theme, visual_html)

        slide_path = slides_dir / f"{chunk_file.replace('.json', '.png')}"

        plan.append({
            "chunk_file": chunk_file,
            "wav_path": wav_path,
            "slide_path": slide_path,
            "html": html,
            "duration": seg["duration"],
        })

    return plan


# ==========================================
# PLAYWRIGHT RENDER
# ==========================================
async def render_slide(page, html: str, out_path: Path):
    await page.set_content(html, wait_until="networkidle")
    await page.wait_for_timeout(500)  # KaTeX render time
    await page.screenshot(path=str(out_path))


# ==========================================
# FFMPEG HELPERS
# ==========================================
def run_ffmpeg(cmd: list, label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFMPEG FAIL [{label}]")
        print(f"    {result.stderr[-500:]}")
        return False
    return True


def ffmpeg_segment(slide: str, audio: str, out: str, duration: float) -> bool:
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-i", slide,
        "-i", audio,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-t", str(duration + 0.1),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-r", str(FPS),
        out
    ], Path(out).name)


def ffmpeg_concat(segment_paths: list, out: Path) -> bool:
    concat_file = out.parent / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy", str(out)
    ], out.name)


def ffmpeg_with_intro_outro(content_video: Path, out: Path) -> bool:
    intro = INTRO_VIDEO
    end = END_VIDEO
    if not intro.exists() or not end.exists():
        print(f"    intro/end not found, copying content as final")
        return run_ffmpeg(["ffmpeg", "-y", "-i", str(content_video), "-c", "copy", str(out)], out.name)
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(intro),
        "-i", str(content_video),
        "-i", str(end),
        "-filter_complex", "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1",
        "-pix_fmt", "yuv420p",
        str(out)
    ], out.name)


# ==========================================
# PROCESS ONE CONCEPT
# ==========================================
async def process_concept(merge_code: str, module_id: str, page):
    chunk_dir = CHUNKS_FOLDER / merge_code / module_id
    audio_dir = AUDIO_FOLDER / merge_code / module_id
    video_dir = VIDEOS_FOLDER / merge_code
    video_dir.mkdir(parents=True, exist_ok=True)
    video_out = video_dir / f"{module_id}.mp4"

    if video_out.exists():
        return "skipped"

    meta_path = chunk_dir / "_meta.json"
    if not meta_path.exists():
        return "no_meta"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    theme = pick_theme(module_id)
    slides_dir = video_dir / f"slides_{module_id}"
    slides_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{theme['name']}] {module_id}")

    # Build slide plan
    plan = build_slide_plan(meta, chunk_dir, audio_dir, theme, slides_dir)

    if not plan:
        print(f"    No slides for {module_id}")
        return "no_slides"

    # Render slides via Playwright
    for entry in plan:
        if entry["slide_path"].exists():
            continue
        await render_slide(page, entry["html"], entry["slide_path"])

    # FFmpeg: create segments
    segments = []
    for entry in plan:
        seg_out = slides_dir / f"seg_{entry['chunk_file'].replace('.json', '.mp4')}"
        if not seg_out.exists():
            ok = ffmpeg_segment(
                str(entry["slide_path"]),
                str(entry["wav_path"]),
                str(seg_out),
                entry["duration"]
            )
            if not ok:
                continue
        segments.append(seg_out)

    if not segments:
        print(f"    No segments for {module_id}")
        return "no_segments"

    # Concat segments
    content_video = slides_dir / "content.mp4"
    if not content_video.exists():
        ok = ffmpeg_concat(segments, content_video)
        if not ok:
            return "concat_failed"

    # Add intro + outro
    ok = ffmpeg_with_intro_outro(content_video, video_out)
    if ok and video_out.exists():
        size_mb = video_out.stat().st_size // 1_000_000
        print(f"    OK {module_id} ({size_mb} MB)")
        return "ok"
    else:
        return "final_failed"


# ==========================================
# MERGE MODULE VIDEO (all concepts for a MERGE_CODE)
# ==========================================
def merge_module_video(merge_code: str):
    video_dir = VIDEOS_FOLDER / merge_code
    merged_out = VIDEOS_FOLDER / f"{merge_code}.mp4"

    if merged_out.exists():
        return

    # Collect concept videos in natural order
    concept_videos = sorted(
        video_dir.glob("*.mp4"),
        key=lambda p: natural_sort_key(p.stem)
    )
    # Filter out working files
    concept_videos = [v for v in concept_videos if not v.stem.startswith("slides_")]

    if len(concept_videos) < 1:
        return

    print(f"  Merging {merge_code}: {len(concept_videos)} concept videos")

    # Concat all concept videos (they already have individual intro/outro,
    # but module merge just concatenates them and wraps with its own intro/outro)
    module_content = video_dir / "module_content.mp4"
    ok = ffmpeg_concat(concept_videos, module_content)
    if not ok:
        print(f"    Merge concat failed for {merge_code}")
        return

    ok = ffmpeg_with_intro_outro(module_content, merged_out)
    if ok and merged_out.exists():
        size_mb = merged_out.stat().st_size // 1_000_000
        print(f"    OK {merge_code}.mp4 ({size_mb} MB)")


def natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally: concept_1 < concept_2 < concept_10."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ==========================================
# FIND READY MODULES
# ==========================================
def get_ready_concepts() -> list[tuple[str, str]]:
    """Find concepts where audio is done but video is not."""
    ready = []
    if not AUDIO_FOLDER.is_dir():
        return ready

    for mc in sorted(AUDIO_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        for mod_dir in sorted(mc.iterdir()):
            if not mod_dir.is_dir():
                continue
            timeline = mod_dir / "timeline.json"
            if not timeline.exists():
                continue

            merge_code = mc.name
            module_id = mod_dir.name

            video_out = VIDEOS_FOLDER / merge_code / f"{module_id}.mp4"
            if video_out.exists():
                continue

            chunk_dir = CHUNKS_FOLDER / merge_code / module_id
            if not (chunk_dir / "_meta.json").exists():
                continue

            ready.append((merge_code, module_id))

    return ready


def get_merge_codes_ready() -> list[str]:
    """Find merge codes where all concepts have videos but module merge is not done."""
    ready = []
    if not CHUNKS_FOLDER.is_dir():
        return ready

    for mc in sorted(CHUNKS_FOLDER.iterdir()):
        if not mc.is_dir():
            continue
        merge_code = mc.name
        merged_out = VIDEOS_FOLDER / f"{merge_code}.mp4"
        if merged_out.exists():
            continue

        video_dir = VIDEOS_FOLDER / merge_code
        if not video_dir.is_dir():
            continue

        concept_dirs = [d for d in sorted(mc.iterdir()) if d.is_dir()]
        if not concept_dirs:
            continue

        all_done = all(
            (video_dir / f"{d.name}.mp4").exists()
            for d in concept_dirs
        )

        if all_done:
            ready.append(merge_code)

    return ready


# ==========================================
# SUMMARY
# ==========================================
def generate_summary():
    print("\n" + "=" * 60)
    print("VIDEO GENERATION SUMMARY")
    print("=" * 60)

    total_concepts = 0
    total_modules = 0
    total_size = 0

    for mc in sorted(VIDEOS_FOLDER.iterdir()):
        if mc.is_file() and mc.suffix == ".mp4":
            total_modules += 1
            total_size += mc.stat().st_size
        elif mc.is_dir():
            for vid in mc.glob("*.mp4"):
                if not vid.stem.startswith("slides_"):
                    total_concepts += 1
                    total_size += vid.stat().st_size

    print(f"  Concept videos: {total_concepts}")
    print(f"  Module videos:  {total_modules}")
    print(f"  Total size:     {total_size // 1_000_000} MB")


# ==========================================
# MAIN
# ==========================================
async def main():
    print("=" * 60)
    print("VIDEO GENERATION PIPELINE")
    print("=" * 60)
    print(f"  Themes:     {len(THEMES)} color themes")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  Intro:      {'YES' if INTRO_VIDEO.exists() else 'NO'}")
    print(f"  Outro:      {'YES' if END_VIDEO.exists() else 'NO'}")
    print()

    empty_polls = 0
    total_processed = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        while True:
            # Phase 1: Concept videos
            ready = get_ready_concepts()

            if DEBUG:
                ready = ready[:1]

            if ready:
                empty_polls = 0
                print(f"\nFound {len(ready)} concepts ready for video")

                for merge_code, module_id in ready:
                    try:
                        await process_concept(merge_code, module_id, page)
                    except Exception as e:
                        print(f"    ERROR on {module_id}: {e}")
                        continue

                total_processed += len(ready)
                print(f"Total processed: {total_processed} concepts")

            # Phase 2: Module merges
            merge_ready = get_merge_codes_ready()
            for mc in merge_ready:
                try:
                    merge_module_video(mc)
                except Exception as e:
                    print(f"    MERGE ERROR on {mc}: {e}")

            if not ready and not merge_ready:
                empty_polls += 1
                if empty_polls >= MAX_EMPTY_POLLS:
                    print("No new work for 30 polls. Finishing.")
                    break
                print(f"No new work. Waiting {POLL_INTERVAL}s... ({empty_polls}/{MAX_EMPTY_POLLS})")
                await asyncio.sleep(POLL_INTERVAL)
                continue

        await browser.close()

    generate_summary()
    print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())
