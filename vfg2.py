import json
import asyncio
import subprocess
import multiprocessing
from pathlib import Path

from playwright.async_api import async_playwright

# ==============================
# CONFIG
# ==============================

CHUNKS_DIR   = Path("chunks_structured")
AUDIO_DIR    = Path("audio")
OUTPUT_DIR   = Path("videos")

INTRO_VIDEO  = "intro_v0.mp4"
END_VIDEO    = "end_v0.mp4"

WIDTH, HEIGHT = 1920, 1080
FPS           = 30
MAX_WORKERS   = 3
DEBUG         = False

print("CPU cores:", multiprocessing.cpu_count())

# ==============================
# DESIGN TOKENS
# ==============================

BRAND   = "#00c6ff"
BG      = "#020202"
TEXT    = "#ffffff"
SUBTEXT = "#cfcfcf"
ACCENT  = "#00c6ff"
STEP_DONE_BG   = "rgba(0,198,255,0.08)"
STEP_ACTIVE_BG = "rgba(0,198,255,0.18)"
STEP_ACTIVE_BORDER = BRAND

# ==============================
# BASE HTML WRAPPER
# injects KaTeX, fonts, global styles
# ==============================

def base_html(body: str) -> str:
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
    background: {BG};
    background-image: radial-gradient(circle at 75% 50%, #001a25 0%, {BG} 65%);
    font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
    color: {TEXT};
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
    letter-spacing: 4px; color: {BRAND};
    text-transform: uppercase;
    border-left: 3px solid {BRAND};
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
  .bullet-list {{ list-style: none; }}
  .bullet-list li {{
    font-size: 36px; line-height: 1.55;
    margin-bottom: 28px; display: flex;
    align-items: flex-start; color: {SUBTEXT};
  }}
  .bullet-node {{
    width: 7px; height: 26px;
    background: {BRAND}; margin-top: 14px;
    margin-right: 24px; flex-shrink: 0;
    border-radius: 1px;
  }}
  /* STEP CARDS */
  .step-card {{
    border-radius: 10px; padding: 18px 24px;
    margin-bottom: 16px;
    border-left: 3px solid transparent;
    font-size: 32px; line-height: 1.5;
    color: rgba(255,255,255,0.35);
    background: transparent;
    transition: all 0.2s;
  }}
  .step-card.done {{
    background: {STEP_DONE_BG};
    border-left-color: rgba(0,198,255,0.3);
    color: rgba(255,255,255,0.55);
  }}
  .step-card.active {{
    background: {STEP_ACTIVE_BG};
    border-left-color: {STEP_ACTIVE_BORDER};
    color: {TEXT};
  }}
  .step-num {{
    font-size: 22px; font-weight: 600;
    color: {BRAND}; margin-bottom: 4px;
  }}
  .justification {{
    font-size: 24px; color: rgba(255,255,255,0.45);
    margin-top: 6px; font-style: italic;
  }}
  /* DEFINITION CARD */
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
  /* FORMULA BOX */
  .formula-box {{
    background: rgba(0,198,255,0.06);
    border: 1px solid rgba(0,198,255,0.25);
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
    font-size: 30px;
  }}
  th {{
    background: rgba(0,198,255,0.15);
    color: {BRAND}; padding: 14px 20px;
    text-align: left; font-weight: 600;
    border-bottom: 1px solid rgba(0,198,255,0.3);
  }}
  td {{
    padding: 12px 20px; color: {SUBTEXT};
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}
  tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
  /* GIVEN TABLE */
  .given-table {{ font-size: 28px; margin-bottom: 28px; }}
  .given-table td:first-child {{
    color: {BRAND}; font-weight: 700; width: 80px;
  }}
  .given-table td:nth-child(2) {{
    color: {TEXT}; font-weight: 600; width: 120px;
  }}
  /* EXAM TIP */
  .exam-tip {{
    background: rgba(255,200,0,0.07);
    border: 1px solid rgba(255,200,0,0.25);
    border-radius: 10px; padding: 20px 26px;
    font-size: 28px; margin-top: 24px;
  }}
  .exam-tip-label {{
    font-size: 20px; font-weight: 700;
    color: #ffc800; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px;
  }}
  /* RECAP GRID */
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
  /* PART LABEL */
  .part-label {{
    font-size: 26px; font-weight: 700;
    color: {BRAND}; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 16px;
    border-bottom: 1px solid rgba(0,198,255,0.2);
    padding-bottom: 10px;
  }}
  /* BOUNDARY CHECK */
  .boundary-box {{
    background: rgba(255,100,100,0.07);
    border: 1px solid rgba(255,100,100,0.25);
    border-radius: 10px; padding: 18px 24px;
    font-size: 28px; margin-top: 20px; color: #ff9999;
  }}
  .boundary-label {{
    font-size: 20px; font-weight: 700;
    color: #ff9999; text-transform: uppercase;
    letter-spacing: 2px; margin-bottom: 6px;
  }}
</style>
</head>
<body>
  <div class="sidebar">
    <div class="brand">SRINIVAS IAS<br>ACADEMY</div>
    <div class="sidebar-meta" id="sidebar-meta"></div>
  </div>
  {body}
</body>
</html>"""

# ==============================
# SIDEBAR META HELPER
# ==============================

def sidebar_meta(chunk: dict, module: dict) -> str:
    return f"""
    <script>
      document.getElementById('sidebar-meta').innerHTML =
        `<div>{module.get('class','')}ನೇ ತರಗತಿ</div>
         <div>ಅಧ್ಯಾಯ {module.get('chapter','')}</div>
         <div style='margin-top:8px;font-size:14px;color:rgba(255,255,255,0.2)'>{chunk.get('chunk_id','')}</div>`;
    </script>"""

# ==============================
# HTML BUILDERS PER CHUNK TYPE
# ==============================

def html_intro(chunk: dict, module: dict) -> str:
    prereqs = ""
    if chunk.get("prerequisites_display"):
        items = "".join(f"<li><span class='bullet-node'></span>{p}</li>"
                        for p in chunk["prerequisites_display"])
        prereqs = f"<div style='font-size:26px;color:rgba(255,255,255,0.4);margin-bottom:12px;text-transform:uppercase;letter-spacing:2px'>ಪೂರ್ವಾಪೇಕ್ಷಿತಗಳು</div><ul class='bullet-list'>{items}</ul>"

    body = f"""
  <div class="content" style="justify-content:center">
    <div style="font-size:26px;color:{BRAND};letter-spacing:3px;text-transform:uppercase;margin-bottom:20px">
      {module.get('domain','')} · {module.get('class','')}ನೇ ತರಗತಿ
    </div>
    <div class="slide-title" style="font-size:68px">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    <div style="font-size:36px;color:{SUBTEXT};max-width:1100px;line-height:1.6">
      {chunk.get('script','')}
    </div>
    <div style="margin-top:48px">{prereqs}</div>
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


def html_definition(chunk: dict, module: dict, visual_html: str = "") -> str:
    bullets = chunk.get("display_bullets") or []
    bullet_items = "".join(
        f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets
    )

    text_col = f"""
    <div class="col-text">
      <ul class="bullet-list">{bullet_items}</ul>
    </div>"""

    visual_col = f'<div class="col-visual">{visual_html}</div>' if visual_html else ""

    layout = "two-col" if visual_html else ""
    inner  = f'<div class="{layout}">{text_col}{visual_col}</div>' if visual_html else text_col

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    {inner}
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


def html_concept_explanation(chunk: dict, module: dict, visual_html: str = "") -> str:
    bullets = chunk.get("display_bullets") or []
    bullet_items = "".join(
        f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets
    )

    text_col = f'<div class="col-text"><ul class="bullet-list">{bullet_items}</ul></div>'
    visual_col = f'<div class="col-visual">{visual_html}</div>' if visual_html else ""

    layout = "two-col" if visual_html else ""
    inner  = f'<div class="{layout}">{text_col}{visual_col}</div>' if visual_html else text_col

    exam_tip_html = build_exam_tip(chunk)

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    {inner}
    {exam_tip_html}
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


def html_formula_derivation(chunk: dict, module: dict, active_step: int) -> str:
    """active_step = 0-based index of currently highlighted step"""
    steps      = chunk.get("derivation_steps") or []
    step_cards = ""
    for i, s in enumerate(steps):
        if i > active_step:
            continue                    # not yet revealed
        state  = "active" if i == active_step else "done"
        action = s.get("action_display") or s.get("action_spoken", "")
        justif = s.get("justification", "")
        justif_html = f'<div class="justification">{justif}</div>' if justif else ""
        step_cards += f"""
        <div class="step-card {state}">
          <div class="step-num">ಹಂತ {s['step']}</div>
          <div>{action}</div>
          {justif_html}
        </div>"""

    result = chunk.get("result_formula") or {}
    formula_box = ""
    if result.get("latex"):
        formula_box = f"""
        <div class="formula-box">
          <div class="formula-label">{result.get('name','')}</div>
          $${result['latex']}$$
        </div>"""

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    <div class="two-col">
      <div class="col-text" style="overflow-y:auto;max-height:780px">{step_cards}</div>
      <div class="col-visual">{formula_box}</div>
    </div>
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


def html_worked_example(chunk: dict, module: dict, active_step: int,
                         active_part: int = 0) -> str:
    """
    active_step = 0-based step index within the active part (or flat steps).
    active_part = 0-based part index for multi-part examples.
    """
    problem   = chunk.get("problem_statement_display") or chunk.get("problem_statement", "")
    parts     = chunk.get("parts")
    flat_steps = chunk.get("solution_steps") or []

    difficulty_badge = ""
    d = chunk.get("difficulty","")
    if d:
        color_map = {"basic": "#44cc88", "intermediate": BRAND, "advanced": "#ff6b6b"}
        col = color_map.get(d, BRAND)
        difficulty_badge = f'<span style="font-size:22px;color:{col};border:1px solid {col};padding:4px 16px;border-radius:20px;margin-left:16px;vertical-align:middle">{d.upper()}</span>'

    step_cards = ""

    if parts:
        part = parts[min(active_part, len(parts)-1)]
        part_label_html = f'<div class="part-label">{part["part_label"]}</div>'
        steps = part.get("solution_steps") or []
        for i, s in enumerate(steps):
            if i > active_step:
                continue
            state  = "active" if i == active_step else "done"
            action = s.get("action_display") or s.get("action_spoken", "")
            justif = s.get("justification","")
            justif_html = f'<div class="justification">{justif}</div>' if justif else ""
            step_cards += f"""
            <div class="step-card {state}">
              <div class="step-num">ಹಂತ {s['step']}</div>
              <div>{action}</div>
              {justif_html}
            </div>"""
        step_cards = part_label_html + step_cards
    else:
        for i, s in enumerate(flat_steps):
            if i > active_step:
                continue
            state  = "active" if i == active_step else "done"
            action = s.get("action_display") or s.get("action_spoken", "")
            justif = s.get("justification","")
            justif_html = f'<div class="justification">{justif}</div>' if justif else ""
            step_cards += f"""
            <div class="step-card {state}">
              <div class="step-num">ಹಂತ {s['step']}</div>
              <div>{action}</div>
              {justif_html}
            </div>"""

    # show final answer only when last step is active
    total_steps = len(parts[active_part]["solution_steps"]) if parts else len(flat_steps)
    final_html = ""
    if active_step >= total_steps - 1:
        fa = chunk.get("final_answer_display") or chunk.get("final_answer","")
        if fa:
            final_html = f"""
            <div style="margin-top:20px;padding:18px 24px;background:rgba(0,198,255,0.12);
                        border-radius:10px;font-size:34px;font-weight:700;color:{BRAND}">
              ✓ &nbsp; {fa}
            </div>"""

    exam_tip_html = build_exam_tip(chunk)

    body = f"""
  <div class="content">
    <div class="slide-title">
      {chunk.get('slide_title','')} {difficulty_badge}
    </div>
    <div class="accent-line"></div>
    <div style="font-size:32px;color:{SUBTEXT};margin-bottom:28px;
                background:rgba(255,255,255,0.04);padding:18px 24px;
                border-radius:8px;border-left:3px solid rgba(0,198,255,0.3)">
      {problem}
    </div>
    <div style="overflow-y:auto;max-height:560px">
      {step_cards}
      {final_html}
    </div>
    {exam_tip_html}
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


def html_length_problem(chunk: dict, module: dict, active_step: int) -> str:
    given      = chunk.get("given") or []
    problem    = chunk.get("problem_statement_display") or chunk.get("problem_statement","")
    flat_steps = chunk.get("solution_steps") or []
    formula    = chunk.get("formula_used") or {}

    given_rows = "".join(
        f"<tr><td>{g['variable']}</td><td>{g['value']}</td><td style='color:rgba(255,255,255,0.4)'>{g['meaning']}</td></tr>"
        for g in given
    )
    given_table = f"""
    <table class="given-table">
      <tr><td colspan='3' style='color:{BRAND};font-weight:700;font-size:22px;
          padding-bottom:8px;text-transform:uppercase;letter-spacing:2px'>ನೀಡಿರುವುದು</td></tr>
      {given_rows}
    </table>""" if given else ""

    formula_html = ""
    if formula.get("latex"):
        formula_html = f"""
        <div class="formula-box" style="margin-bottom:24px;font-size:34px">
          <div class="formula-label">{formula.get('name','')}</div>
          $${formula['latex']}$$
        </div>"""

    step_cards = ""
    for i, s in enumerate(flat_steps):
        if i > active_step:
            continue
        state  = "active" if i == active_step else "done"
        action = s.get("action_display") or s.get("action_spoken","")
        justif = s.get("justification","")
        justif_html = f'<div class="justification">{justif}</div>' if justif else ""
        step_cards += f"""
        <div class="step-card {state}">
          <div class="step-num">ಹಂತ {s['step']}</div>
          <div>{action}</div>
          {justif_html}
        </div>"""

    boundary_html = ""
    if active_step >= len(flat_steps) - 1:
        bc = chunk.get("boundary_check_display") or chunk.get("boundary_check","")
        fa = chunk.get("final_answer_display") or chunk.get("final_answer","")
        if fa:
            boundary_html += f"""
            <div style="padding:18px 24px;background:rgba(0,198,255,0.12);
                        border-radius:10px;font-size:34px;font-weight:700;
                        color:{BRAND};margin-top:16px">
              ✓ &nbsp; {fa}
            </div>"""
        if bc:
            boundary_html += f"""
            <div class="boundary-box">
              <div class="boundary-label">⚠ ಮಿತಿ ಪರಿಶೀಲನೆ</div>
              {bc}
            </div>"""

    exam_tip_html = build_exam_tip(chunk)

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title','')}</div>
    <div class="accent-line"></div>
    <div class="two-col">
      <div class="col-text">
        <div style="font-size:30px;color:{SUBTEXT};margin-bottom:20px;
                    background:rgba(255,255,255,0.04);padding:16px 22px;
                    border-radius:8px;border-left:3px solid rgba(0,198,255,0.3)">
          {problem}
        </div>
        {step_cards}
        {boundary_html}
        {exam_tip_html}
      </div>
      <div class="col-visual">
        {given_table}
        {formula_html}
      </div>
    </div>
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


def html_recap(chunk: dict, module: dict) -> str:
    cs = chunk.get("coverage_summary") or {}
    cells = [
        ("ವ್ಯಾಖ್ಯೆಗಳು",     cs.get("definitions_covered", 0)),
        ("ಸೂತ್ರಗಳು",        cs.get("formulas_covered", 0)),
        ("ಪ್ರಮೇಯಗಳು",       cs.get("theorems_covered", 0)),
        ("ಗುಣಧರ್ಮಗಳು",      cs.get("properties_covered", 0)),
        ("ಉದಾಹರಣೆಗಳು",      cs.get("worked_examples_covered", 0)),
        ("ಉದ್ದದ ಸಮಸ್ಯೆಗಳು", cs.get("length_problems_covered", 0)),
    ]
    grid = "".join(
        f'<div class="recap-cell"><div class="recap-count">{v}</div>'
        f'<div class="recap-label">{k}</div></div>'
        for k, v in cells
    )

    bullets = chunk.get("display_bullets") or []
    bullet_items = "".join(
        f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets
    )

    next_mods = chunk.get("next_modules") or []
    next_html = ""
    if next_mods:
        items = " · ".join(next_mods)
        next_html = f"""
        <div style="margin-top:28px;font-size:24px;color:rgba(255,255,255,0.35)">
          ಮುಂದೆ: <span style="color:{BRAND}">{items}</span>
        </div>"""

    body = f"""
  <div class="content">
    <div class="slide-title">{chunk.get('slide_title', 'ಪುನರಾವರ್ತನೆ')}</div>
    <div class="accent-line"></div>
    <div class="two-col">
      <div class="col-text">
        <ul class="bullet-list">{bullet_items}</ul>
        {next_html}
      </div>
      <div class="col-visual">
        <div class="recap-grid">{grid}</div>
      </div>
    </div>
  </div>
  {sidebar_meta(chunk, module)}"""
    return base_html(body)


# ==============================
# VISUAL HTML BUILDER
# renders table visuals inline
# diagram types are placeholders (plotly/svg rendered separately if needed)
# ==============================

def build_visual_html(visual: dict) -> str:
    if not visual or visual.get("type") == "none":
        return ""

    vtype = visual.get("type","")

    if vtype == "table":
        headers = visual.get("headers") or []
        rows    = visual.get("rows") or []
        th_html = "".join(f"<th>{h}</th>" for h in headers)
        tr_html = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in rows
        )
        purpose = visual.get("purpose","")
        purpose_html = f'<div style="font-size:22px;color:rgba(255,255,255,0.4);margin-bottom:12px">{purpose}</div>' if purpose else ""
        return f"{purpose_html}<table><thead><tr>{th_html}</tr></thead><tbody>{tr_html}</tbody></table>"

    if vtype == "formula_box":
        latex = visual.get("description","") or visual.get("concept_latex","")
        return f'<div class="formula-box">$${latex}$$</div>' if latex else ""

    if vtype == "diagram":
        desc = visual.get("description","") or visual.get("graph_type","diagram")
        return f"""
        <div style="border:1px dashed rgba(0,198,255,0.3);border-radius:10px;
                    padding:40px;text-align:center;color:rgba(255,255,255,0.3);
                    font-size:24px">
          [{desc}]<br><span style="font-size:18px">render: {visual.get('render_target','')}</span>
        </div>"""

    return ""


def build_exam_tip(chunk: dict) -> str:
    et = chunk.get("exam_tip")
    if not et:
        return ""
    return f"""
    <div class="exam-tip">
      <div class="exam-tip-label">💡 ಪರೀಕ್ಷಾ ಸೂಚನೆ · {et.get('question_pattern','')}</div>
      <div>{et.get('skill_tested','')}</div>
      <div style="margin-top:6px;color:rgba(255,200,0,0.6);font-size:24px">
        ⚠ {et.get('distractor','')}
      </div>
    </div>"""


# ==============================
# RENDER ONE SLIDE TO PNG
# ==============================

async def render_slide(page, html: str, out_path: Path):
    await page.set_content(html, wait_until="networkidle")
    await page.wait_for_timeout(300)   # let KaTeX render
    await page.screenshot(path=str(out_path))


# ==============================
# SLIDE PLAN
# maps each timeline label → (html_string, slide_path)
# ==============================

def build_slide_plan(chunk_file_data: dict, slides_dir: Path) -> list[dict]:
    """
    Returns a list of dicts in timeline label order:
      { label, slide_path, html }
    For per_step chunks: one slide per step (step reveals accumulate)
    For chunk-mode chunks: one slide for the whole chunk
    """
    module   = chunk_file_data
    chunks   = module.get("chunks", [])
    plan     = []

    for chunk in chunks:
        ctype     = chunk.get("type","")
        sync_mode = chunk.get("tts", {}).get("sync_mode", "chunk")
        visual    = chunk.get("visual") or {}
        vis_html  = build_visual_html(visual)
        chunk_id  = chunk.get("chunk_id","")

        if ctype == "intro":
            label = f"{chunk_id}_script"
            path  = slides_dir / f"{label}.png"
            plan.append({"label": label, "slide_path": path,
                         "html": html_intro(chunk, module)})

        elif ctype == "definition":
            label = f"{chunk_id}_script"
            path  = slides_dir / f"{label}.png"
            plan.append({"label": label, "slide_path": path,
                         "html": html_definition(chunk, module, vis_html)})

        elif ctype == "concept_explanation":
            label = f"{chunk_id}_script"
            path  = slides_dir / f"{label}.png"
            plan.append({"label": label, "slide_path": path,
                         "html": html_concept_explanation(chunk, module, vis_html)})

        elif ctype == "formula_derivation":
            steps = chunk.get("derivation_steps") or []
            for i, s in enumerate(steps):
                label = f"{chunk_id}_deriv_{s['step']}"
                path  = slides_dir / f"{label}.png"
                plan.append({"label": label, "slide_path": path,
                             "html": html_formula_derivation(chunk, module, i)})

        elif ctype == "worked_example":
            parts      = chunk.get("parts")
            flat_steps = chunk.get("solution_steps") or []

            if parts:
                for pi, part in enumerate(parts):
                    for si, s in enumerate(part.get("solution_steps",[])):
                        label = f"{chunk_id}_{part['part_label']}_step_{s['step']}"
                        path  = slides_dir / f"{label}.png"
                        plan.append({"label": label, "slide_path": path,
                                     "html": html_worked_example(chunk, module, si, pi)})
            else:
                for i, s in enumerate(flat_steps):
                    label = f"{chunk_id}_step_{s['step']}"
                    path  = slides_dir / f"{label}.png"
                    plan.append({"label": label, "slide_path": path,
                                 "html": html_worked_example(chunk, module, i)})

        elif ctype == "length_problem":
            steps = chunk.get("solution_steps") or []
            for i, s in enumerate(steps):
                label = f"{chunk_id}_step_{s['step']}"
                path  = slides_dir / f"{label}.png"
                plan.append({"label": label, "slide_path": path,
                             "html": html_length_problem(chunk, module, i)})

        elif ctype == "recap":
            label = f"{chunk_id}_script"
            path  = slides_dir / f"{label}.png"
            plan.append({"label": label, "slide_path": path,
                         "html": html_recap(chunk, module)})

    return plan


# ==============================
# GENERATE ALL SLIDES (Playwright)
# ==============================

async def generate_slides(chunk_file_data: dict, slides_dir: Path):
    slides_dir.mkdir(parents=True, exist_ok=True)
    plan = build_slide_plan(chunk_file_data, slides_dir)

    # skip if all slides already exist
    if all(p["slide_path"].exists() for p in plan):
        print(f"  ⏭ Slides already exist")
        return plan

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page    = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for entry in plan:
            if entry["slide_path"].exists():
                continue
            print(f"  🖼 {entry['slide_path'].name}")
            await render_slide(page, entry["html"], entry["slide_path"])

        await browser.close()

    return plan


# ==============================
# FFMPEG HELPERS — full debug
# ==============================

def run_ffmpeg(cmd: list, label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ ffmpeg FAILED [{label}]")
        print(f"     CMD: {' '.join(str(c) for c in cmd[:10])}...")
        print(f"     ERR:\n{result.stderr[-1000:]}")
        return False
    return True


def ffmpeg_segment(slide: str, audio: str, out: str, duration: float) -> bool:
    print(f"  🎬 encoding segment: {Path(out).name}")
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
    with open(concat_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
    print(f"  🔗 concat {len(segment_paths)} segments → {out.name}")
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy", str(out)
    ], out.name)
    if ok:
        print(f"     concat OK  size={out.stat().st_size//1000}KB")
    return ok


def ffmpeg_with_intro_end(content_video: Path, out: Path) -> bool:
    intro = Path(INTRO_VIDEO)
    end   = Path(END_VIDEO)
    if not intro.exists() or not end.exists():
        print(f"  ⚠  intro/end videos not found — copying content directly to {out.name}")
        return run_ffmpeg(["ffmpeg", "-y", "-i", str(content_video), "-c", "copy", str(out)], out.name)
    print(f"  🎞  adding intro + end → {out.name}")
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(intro),
        "-i", str(content_video),
        "-i", str(end),
        "-filter_complex", "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1",
        "-pix_fmt", "yuv420p",
        str(out)
    ], out.name)


# ==============================
# ASSEMBLE MODULE VIDEO
# ==============================

def assemble_video(plan: list, timeline: list, slides_dir: Path,
                   output_path: Path, audio_dir: Path):
    """
    timeline entries have: label, text, start, end, duration  (NO file field)
    audio files are seg_0000.wav, seg_0001.wav ... in audio_dir
    map by index: timeline[i] → audio_dir/seg_{i:04d}.wav
    """
    print(f"\n  📋 assemble_video: {len(plan)} slides, {len(timeline)} timeline entries")
    print(f"  📂 audio_dir: {audio_dir}")

    # build label → (timeline_entry, seg_wav_path) by index
    tl_map = {}
    for i, t in enumerate(timeline):
        wav = audio_dir / f"seg_{i:04d}.wav"
        tl_map[t["label"]] = (t, wav)

    print(f"  📌 plan labels (first 3):     {[e['label'] for e in plan[:3]]}")
    print(f"  📌 timeline labels (first 3): {[t['label'] for t in timeline[:3]]}")

    segments = []
    for entry in plan:
        label = entry["label"]
        match = tl_map.get(label)

        if not match:
            print(f"  ⚠  NO MATCH in timeline for label: [{label}]")
            continue

        tl, audio_path = match
        slide_path = entry["slide_path"]

        if not audio_path.exists():
            print(f"  ⚠  Audio WAV missing: {audio_path}")
            continue
        if not slide_path.exists():
            print(f"  ⚠  Slide PNG missing: {slide_path}")
            continue

        # sanitise label for filename (Kannada chars safe on Linux but colons/spaces are not)
        safe_label = label.replace(" ", "_").replace(":", "-").replace("/", "-")
        seg_out = slides_dir / f"seg_{safe_label}.mp4"

        if not seg_out.exists():
            ok = ffmpeg_segment(str(slide_path), str(audio_path), str(seg_out), tl["duration"])
            if not ok:
                continue
        else:
            print(f"  ⏭  seg exists: {seg_out.name}")
        segments.append(seg_out)

    print(f"\n  ✅ {len(segments)}/{len(plan)} segments ready")

    if not segments:
        print("  ❌ Zero segments — nothing to assemble.")
        return

    content_video = slides_dir / "module_content.mp4"
    ok = ffmpeg_concat(segments, content_video)
    if not ok or not content_video.exists():
        print("  ❌ Concat failed")
        return

    ffmpeg_with_intro_end(content_video, output_path)
    if output_path.exists():
        print(f"  ✅ DONE: {output_path}  ({output_path.stat().st_size // 1_000_000} MB)")
    else:
        print(f"  ❌ Final MP4 not created: {output_path}")


# ==============================
# COLLECT JOBS
# ==============================

def collect_jobs() -> list[tuple]:
    jobs = []

    for chunk_file in sorted(CHUNKS_DIR.rglob("*.json")):
        relative      = chunk_file.relative_to(CHUNKS_DIR)
        module_id     = chunk_file.stem
        timeline_file = AUDIO_DIR / relative.parent / module_id / "timeline.json"
        output_path   = OUTPUT_DIR / relative.parent / f"{module_id}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not timeline_file.exists():
            print(f"⚠  Timeline missing for {module_id}, skipping")
            continue

        jobs.append((chunk_file, timeline_file, output_path))

    return jobs


# ==============================
# MAIN — single async loop, one browser instance
# NO ProcessPoolExecutor — asyncio.run inside workers
# silently hangs on Linux with Playwright
# ==============================

async def main():
    print("\n🎬 Video Pipeline Started\n")

    jobs = collect_jobs()

    if not jobs:
        print("No jobs found. Check chunks_structured/ and audio/ exist.")
        return

    if DEBUG:
        jobs = jobs[:1]

    print(f"📦 {len(jobs)} modules to render\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page    = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for idx, (chunk_file, timeline_file, output_path) in enumerate(jobs, 1):

            if output_path.exists():
                print(f"⏭  [{idx}/{len(jobs)}] Skipped: {output_path.name}")
                continue

            print(f"\n📦 [{idx}/{len(jobs)}] {chunk_file.stem}")

            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)

                with open(timeline_file, "r", encoding="utf-8") as f:
                    timeline = json.load(f)

                slides_dir = output_path.parent / f"slides_{chunk_file.stem}"
                slides_dir.mkdir(parents=True, exist_ok=True)

                # audio_dir = folder containing timeline.json and seg_XXXX.wav
                audio_dir = timeline_file.parent

                # generate slides
                plan = build_slide_plan(chunk_data, slides_dir)

                for entry in plan:
                    if entry["slide_path"].exists():
                        continue
                    print(f"  🖼  {entry['slide_path'].name}")
                    await render_slide(page, entry["html"], entry["slide_path"])

                # assemble with ffmpeg
                assemble_video(plan, timeline, slides_dir, output_path, audio_dir)

                print(f"  ✅ Saved: {output_path}")

            except Exception as e:
                print(f"  ❌ Error on {chunk_file.stem}: {e}")
                continue

        await browser.close()

    print("\n🎉 All videos completed")


if __name__ == "__main__":
    asyncio.run(main())