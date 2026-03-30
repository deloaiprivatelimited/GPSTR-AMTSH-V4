"""
generate_video_v2.py
---------------------
New chunk structure -> Playwright slides -> FFmpeg video

- Reads per-chunk JSON files from chunks_structured/{merge_code}/{module_id}/
- Uses _meta.json for ordering, timeline.json / _audio_meta.json for audio mapping
- Distinct HTML template per layout_config.layout
- ONE video per chapter (merge_code) -- all modules concatenated in order
- Audio from audio_v2/{merge_code}/{module_id}/
"""

import json
import asyncio
import subprocess
import multiprocessing
import wave
from pathlib import Path

from playwright.async_api import async_playwright

# ==============================
# CONFIG
# ==============================

CHUNKS_DIR = Path("claude_works/chunks_structured")
AUDIO_DIR  = Path("claude_works/audio_v2")
OUTPUT_DIR = Path("claude_works/videos_v2")

INTRO_VIDEO = Path("intro_v0.mp4")
END_VIDEO   = Path("end_v0.mp4")

WIDTH, HEIGHT = 1920, 1080
FPS           = 30
DEBUG         = False

print("CPU cores:", multiprocessing.cpu_count())

# ==============================
# DESIGN TOKENS
# ==============================

BRAND   = "#00c6ff"
BG      = "#020202"
TEXT    = "#ffffff"
SUBTEXT = "#cfcfcf"

# ==============================
# GLOBAL STYLES (shared by all layouts)
# ==============================

GLOBAL_STYLES = f"""
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

  /* -- SIDEBAR -- */
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

  /* -- CONTENT AREA -- */
  .content {{
    flex-grow: 1;
    padding: 70px 80px 70px 70px;
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
    font-size: 28px;
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

def base_html(body, sidebar_html=""):
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
  <div class="sidebar">
    <div class="brand">SRINIVAS IAS<br>ACADEMY</div>
    <div class="sidebar-meta">{sidebar_html}</div>
  </div>
  {body}
</body>
</html>"""


# ==============================
# HELPERS
# ==============================

def make_sidebar(chunk, meta):
    cls = meta.get("class", "")
    ch  = meta.get("chapter", "")
    cid = chunk.get("chunk_id", "")
    return (f"<div>{cls}ನೇ ತರಗತಿ</div>"
            f"<div>ಅಧ್ಯಾಯ {ch}</div>"
            f"<div style='margin-top:8px;font-size:14px;color:rgba(255,255,255,0.2)'>{cid}</div>")


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
        th = "".join(f"<th>{h}</th>" for h in headers)
        tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        caption = visual.get("caption", "")
        cap_html = f'<div style="font-size:20px;color:rgba(255,255,255,0.4);margin-bottom:12px">{caption}</div>' if caption else ""
        return f"{cap_html}<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"

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
        inner = f"""<div class="two-col">
            <div class="col-text">{bullets}</div>
            {vis_col}
        </div>"""
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
        inner = f"""<div class="two-col">
            <div class="col-text">{bullets}</div>
            <div class="col-visual">{vis}</div>
        </div>"""
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
            <div class="col-text" style="overflow-y:auto;max-height:780px">
              {text_content}{final_html}
            </div>
            <div class="col-visual">{vis}</div>
        </div>"""
    else:
        inner = f"""<div style="overflow-y:auto;max-height:780px">
            {text_content}{final_html}
        </div>"""

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
        inner = f"""<div class="two-col">
            <div class="col-text">{bullets}</div>
            <div class="col-visual">{vis}</div>
        </div>"""
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
        ("Definitions",      cs.get("definitions", 0)),
        ("Formulas",         cs.get("formulas", 0)),
        ("Theorems",         cs.get("theorems", 0)),
        ("Properties",       cs.get("properties", 0)),
        ("Worked Examples",  cs.get("worked_examples", 0)),
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
        <div class="col-text">
          {bullets}
          {next_html}
        </div>
        <div class="col-visual">
          <div class="recap-grid">{grid}</div>
        </div>
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
    sidebar_html = make_sidebar(chunk, meta)
    return base_html(content_html, sidebar_html)


# ==============================
# RENDER SLIDE TO PNG
# ==============================

async def render_slide(page, html, out_path):
    await page.set_content(html, wait_until="networkidle")
    await page.wait_for_timeout(500)  # KaTeX render time
    await page.screenshot(path=str(out_path))


# ==============================
# COLLECT ALL CHUNKS FOR A CHAPTER
# ==============================

def get_wav_duration(wav_path):
    """Read actual duration from WAV file header -- ground truth."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def collect_chapter_chunks(merge_code):
    """
    Collects all chunks across all modules for a merge_code (chapter),
    ordered by module then by chunk_order within each module.
    Duration is read directly from the WAV file (ground truth).
    """
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

        # Skip modules with chunk generation errors
        if meta.get("errors"):
            print(f"  SKIP {module_id}: has chunk errors")
            continue

        # Skip if timeline.json doesn't exist -- means audio is not complete
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

            # Audio: same name, .wav instead of .json
            wav_name = chunk_file.replace(".json", ".wav")
            audio_path = audio_base / module_id / wav_name

            # Duration from WAV file directly
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
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-t", str(duration + 0.1),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-r", str(FPS),
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
# ASSEMBLE CHAPTER VIDEO
# ==============================

def assemble_chapter_video(entries, slides_dir, output_path):
    segments = []

    for entry in entries:
        slide_path = entry.get("slide_path")
        audio_path = entry["audio_path"]
        duration   = entry["duration"]

        if not slide_path or not Path(slide_path).exists():
            print(f"  WARN slide missing: {entry['chunk_file']}")
            continue
        if not audio_path.exists():
            print(f"  WARN audio missing: {audio_path}")
            continue
        if duration <= 0:
            continue

        safe_name = entry["chunk_file"].replace(".json", "").replace(" ", "_")
        seg_out = slides_dir / f"seg_{entry['module_id']}_{safe_name}.mp4"

        if not seg_out.exists():
            ok = ffmpeg_segment(str(slide_path), str(audio_path), str(seg_out), duration)
            if not ok:
                continue
        segments.append(seg_out)

    print(f"\n  {len(segments)}/{len(entries)} segments ready")

    if not segments:
        print("  Zero segments -- nothing to assemble.")
        return

    content_video = slides_dir / "chapter_content.mp4"
    ok = ffmpeg_concat(segments, content_video)
    if not ok or not content_video.exists():
        print("  Concat failed")
        return

    ffmpeg_with_intro_end(content_video, output_path)
    if output_path.exists():
        size_mb = output_path.stat().st_size // 1_000_000
        print(f"  DONE: {output_path}  ({size_mb} MB)")
    else:
        print(f"  Final MP4 not created: {output_path}")


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
    print("\nVideo Pipeline v2 -- per-chapter\n")

    merge_codes = collect_jobs()
    if not merge_codes:
        print("No chapters found in", CHUNKS_DIR)
        return

    if DEBUG:
        merge_codes = merge_codes[:1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(merge_codes)} chapters to render\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page    = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for idx, merge_code in enumerate(merge_codes, 1):
            output_path = OUTPUT_DIR / f"{merge_code}.mp4"

            if output_path.exists():
                print(f"SKIP [{idx}/{len(merge_codes)}]: {output_path.name}")
                continue

            print(f"\n[{idx}/{len(merge_codes)}] Chapter: {merge_code}")

            try:
                entries = collect_chapter_chunks(merge_code)
                if not entries:
                    print(f"  No chunks found for {merge_code}")
                    continue

                print(f"  {len(entries)} slides")

                slides_dir = OUTPUT_DIR / f"slides_{merge_code}"
                slides_dir.mkdir(parents=True, exist_ok=True)

                # Render all slides to PNG
                for entry in entries:
                    safe_name = entry["chunk_file"].replace(".json", "")
                    slide_name = f"{entry['module_id']}_{safe_name}.png"
                    slide_path = slides_dir / slide_name
                    entry["slide_path"] = slide_path

                    if slide_path.exists():
                        continue

                    html = render_chunk_html(entry["chunk_data"], entry["meta"])
                    print(f"  Slide: {slide_name}")
                    await render_slide(page, html, slide_path)

                # Assemble video
                assemble_chapter_video(entries, slides_dir, output_path)

            except Exception as e:
                print(f"  ERROR on {merge_code}: {e}")
                import traceback
                traceback.print_exc()
                continue

        await browser.close()

    print("\nAll chapter videos completed")


if __name__ == "__main__":
    asyncio.run(main())
