"""
design_v3_sample.py
--------------------
White + Orange theme design -- standalone sample frame generator.
Reads chunk JSONs and renders sample PNGs for approval.
"""

import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

WIDTH, HEIGHT = 1920, 1080
CHUNKS_DIR = Path("claude_works/chunks_structured")
OUTPUT_DIR = Path("claude_works/sample_frames_v3")

# ── DESIGN TOKENS ──
ACCENT   = "#E8651A"
ACCENT2  = "#F28C28"
BG       = "#FFFFFF"
TEXT     = "#1A1A1A"
SUBTEXT  = "#4A4A4A"
MUTED    = "#888888"
LIGHT_BG = "#FFF7F0"
CARD_BG  = "#FEF0E5"
BORDER   = "#F0D4BC"

STYLES = f"""
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Kannada:wght@400;500;700&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    width: {WIDTH}px; height: {HEIGHT}px;
    background: {BG};
    font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
    color: {TEXT};
    display: flex; flex-direction: column;
    overflow: hidden;
  }}

  /* ── HEADER ── */
  .header {{
    width: 100%; height: 56px;
    display: flex; align-items: center;
    padding: 0 64px;
    background: {ACCENT};
    flex-shrink: 0;
  }}
  .brand {{
    font-size: 15px; font-weight: 700;
    letter-spacing: 3px; color: #fff;
    text-transform: uppercase;
  }}
  .header-meta {{
    margin-left: auto;
    font-size: 13px; font-weight: 500;
    color: rgba(255,255,255,0.8);
  }}

  /* ── CONTENT ── */
  .content {{
    flex: 1;
    padding: 48px 72px 48px 72px;
    display: flex; flex-direction: column;
    justify-content: center;
    overflow: hidden;
  }}

  .slide-title {{
    font-size: 46px; font-weight: 800;
    line-height: 1.25; margin-bottom: 10px;
    color: {TEXT};
  }}
  .accent-line {{
    width: 80px; height: 4px;
    background: {ACCENT};
    margin-bottom: 36px; border-radius: 2px;
  }}

  .two-col {{
    display: flex; gap: 56px;
    align-items: flex-start; flex-grow: 1;
  }}
  .col-text {{ flex: 1; }}
  .col-visual {{ flex: 0 0 500px; }}

  /* ── BULLETS ── */
  .bullet-list {{ list-style: none; }}
  .bullet-list li {{
    font-size: 30px; line-height: 1.55;
    margin-bottom: 20px; display: flex;
    align-items: flex-start; color: {SUBTEXT};
  }}
  .bullet-node {{
    width: 6px; height: 24px;
    background: {ACCENT}; margin-top: 10px;
    margin-right: 22px; flex-shrink: 0;
    border-radius: 1px;
  }}

  /* ── STEP CARDS ── */
  .step-card {{
    border-radius: 10px; padding: 18px 24px;
    margin-bottom: 14px;
    border-left: 4px solid {ACCENT2};
    font-size: 28px; line-height: 1.5;
    color: {SUBTEXT};
    background: {LIGHT_BG};
  }}
  .step-card.active {{
    background: {CARD_BG};
    border-left-color: {ACCENT};
    color: {TEXT};
  }}
  .step-num {{
    font-size: 18px; font-weight: 700;
    color: {ACCENT}; margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  .step-justification {{
    font-size: 20px; color: {MUTED};
    margin-top: 4px; font-style: italic;
  }}

  /* ── FORMULA BOX ── */
  .formula-box {{
    background: {LIGHT_BG};
    border: 2px solid {BORDER};
    border-radius: 14px; padding: 28px 36px;
    text-align: center; font-size: 42px;
    color: {TEXT};
    margin-bottom: 24px;
  }}
  .formula-label {{
    font-size: 18px; color: {MUTED};
    margin-bottom: 12px; text-transform: uppercase;
    letter-spacing: 2px; font-weight: 600;
  }}

  /* ── TABLE ── */
  table {{
    border-collapse: collapse; width: 100%;
    font-size: 24px; table-layout: fixed;
  }}
  th {{
    background: {ACCENT};
    color: #fff; padding: 14px 20px;
    text-align: left; font-weight: 600;
    border-bottom: 2px solid {ACCENT};
    word-wrap: break-word; overflow-wrap: break-word;
  }}
  td {{
    padding: 12px 20px; color: {SUBTEXT};
    border-bottom: 1px solid #eee;
    word-wrap: break-word; overflow-wrap: break-word;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: {LIGHT_BG}; }}

  /* ── COMPARE COLS ── */
  .compare-col {{
    flex: 1; padding: 28px;
    border-radius: 14px;
  }}
  .compare-col.left {{
    background: {LIGHT_BG};
    border: 2px solid {BORDER};
  }}
  .compare-col.right {{
    background: #FFF5F5;
    border: 2px solid #F5C6C6;
  }}
  .compare-title {{
    font-size: 26px; font-weight: 700;
    margin-bottom: 16px;
  }}
  .compare-col.left .compare-title {{ color: {ACCENT}; }}
  .compare-col.right .compare-title {{ color: #CC4444; }}

  /* ── RECAP GRID ── */
  .recap-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px; margin-top: 8px;
  }}
  .recap-cell {{
    background: {LIGHT_BG};
    border: 2px solid {BORDER};
    border-radius: 12px; padding: 22px;
    text-align: center;
  }}
  .recap-count {{
    font-size: 50px; font-weight: 800;
    color: {ACCENT}; line-height: 1;
  }}
  .recap-label {{
    font-size: 20px; color: {SUBTEXT};
    margin-top: 6px;
  }}

  /* ── TAKEAWAY ── */
  .takeaway-box {{
    background: {CARD_BG};
    border: 2px solid {ACCENT2};
    border-radius: 14px; padding: 34px 40px;
    font-size: 34px; line-height: 1.6;
    color: {TEXT}; text-align: center;
  }}

  /* ── EXAM TIP ── */
  .exam-tip {{
    background: #FFFDE6;
    border: 1px solid #E6D96B;
    border-radius: 10px; padding: 18px 24px;
    font-size: 24px; margin-top: 20px;
  }}
  .exam-tip-label {{
    font-size: 16px; font-weight: 700;
    color: #B8960C; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 6px;
  }}

  /* ── ANSWER BOX ── */
  .answer-box {{
    margin-top: 18px; padding: 18px 24px;
    background: {CARD_BG};
    border-radius: 10px; font-size: 32px;
    font-weight: 700; color: {ACCENT};
    border: 2px solid {ACCENT2};
  }}

  /* ── BADGE ── */
  .badge {{
    font-size: 18px; padding: 4px 14px;
    border-radius: 20px; margin-left: 14px;
    vertical-align: middle; display: inline-block;
    font-weight: 600;
  }}
  .badge-basic       {{ color: #2E8B57; border: 1.5px solid #2E8B57; }}
  .badge-intermediate {{ color: {ACCENT}; border: 1.5px solid {ACCENT}; }}
  .badge-advanced     {{ color: #CC4444; border: 1.5px solid #CC4444; }}

  /* ── DIAGRAM PLACEHOLDER ── */
  .diagram-placeholder {{
    border: 2px dashed {BORDER};
    border-radius: 12px; padding: 40px;
    text-align: center; color: {MUTED};
    font-size: 20px; background: {LIGHT_BG};
  }}

  /* ── PROBLEM BOX ── */
  .problem-box {{
    font-size: 28px; color: {SUBTEXT};
    background: {LIGHT_BG};
    padding: 18px 24px; border-radius: 10px;
    border-left: 4px solid {ACCENT};
    margin-bottom: 24px; line-height: 1.6;
  }}
"""


# ── HTML WRAPPER ──

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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Kannada:wght@400;500;700&display=swap" rel="stylesheet">
<style>{STYLES}</style>
</head>
<body>
  <div class="header">
    <div class="brand">Srinivas IAS Academy</div>
    <div class="header-meta">{header_meta}</div>
  </div>
  {body}
</body>
</html>"""


# ── HELPERS ──

def make_header_meta(meta):
    cls = meta.get("class", "")
    ch  = meta.get("chapter", "")
    return f"{cls}ನೇ ತರಗತಿ &nbsp;|&nbsp; ಅಧ್ಯಾಯ {ch}"


def make_bullets(bullets):
    if not bullets:
        return ""
    items = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets)
    return f'<ul class="bullet-list">{items}</ul>'


def make_visual(visual):
    if not visual or visual.get("type") == "none":
        return ""
    vtype = visual.get("type", "")

    if vtype == "table":
        headers = visual.get("headers") or []
        rows = visual.get("rows") or []
        num_cols = len(headers) if headers else (len(rows[0]) if rows else 1)
        fs = "20px" if num_cols > 4 else ("22px" if num_cols > 3 else "24px")
        th = "".join(f"<th>{h}</th>" for h in headers)
        tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        cap = visual.get("caption", "")
        cap_html = f'<div style="font-size:18px;color:{MUTED};margin-bottom:10px">{cap}</div>' if cap else ""
        return f"{cap_html}<table style='font-size:{fs}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"

    if vtype == "formula_box":
        latex = visual.get("latex", "")
        desc = visual.get("description", "")
        label = f'<div class="formula-label">{desc}</div>' if desc else ""
        return f'<div class="formula-box">{label}$${latex}$$</div>' if latex else ""

    if vtype == "diagram":
        desc = visual.get("description", "diagram")
        sig = visual.get("mathematical_significance", "")
        return (f'<div class="diagram-placeholder">'
                f'[{desc}]'
                f'<br><span style="font-size:15px">{sig}</span>'
                f'</div>')
    return ""


def make_exam_tip(chunk):
    et = chunk.get("exam_tip")
    if not et:
        return ""
    return f"""
    <div class="exam-tip">
      <div class="exam-tip-label">EXAM TIP - {et.get('question_pattern','')}</div>
      <div>{et.get('skill_tested','')}</div>
    </div>"""


def make_steps(steps):
    if not steps:
        return ""
    cards = ""
    for s in steps:
        action = s.get("action_display", "")
        justif = s.get("justification", "")
        jhtml = f'<div class="step-justification">{justif}</div>' if justif else ""
        cards += f"""
        <div class="step-card">
          <div class="step-num">Step {s.get('step','')}</div>
          <div>{action}</div>
          {jhtml}
        </div>"""
    return cards


def difficulty_badge(d):
    if not d:
        return ""
    cls = f"badge-{d}" if d in ("basic","intermediate","advanced") else "badge-intermediate"
    return f'<span class="badge {cls}">{d.upper()}</span>'


# ── LAYOUTS ──

def layout_title_hero(chunk, meta):
    domain = meta.get("domain", "")
    cls = meta.get("class", "")
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))

    subtitle = f"""<div style="font-size:22px;color:{ACCENT};letter-spacing:2px;
                   text-transform:uppercase;margin-bottom:16px;font-weight:600">
                   {domain} - {cls}ನೇ ತರಗತಿ</div>""" if domain else ""

    vis_col = f'<div class="col-visual">{vis}</div>' if vis else ""
    if vis:
        inner = f'<div class="two-col"><div class="col-text">{bullets}</div>{vis_col}</div>'
    else:
        inner = bullets

    return f"""
    <div class="content" style="justify-content:center">
      {subtitle}
      <div class="slide-title" style="font-size:56px">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {inner}
    </div>"""


def layout_definition_spotlight(chunk, meta):
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))
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
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))
    if not vis:
        v = chunk.get("visual", {})
        if v.get("latex"):
            vis = f'<div class="formula-box" style="font-size:46px">$${v["latex"]}$$</div>'
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      <div class="two-col">
        <div class="col-text">{bullets}</div>
        <div class="col-visual">{vis}</div>
      </div>
      {make_exam_tip(chunk)}
    </div>"""


def layout_step_walkthrough(chunk, meta):
    steps = make_steps(chunk.get("steps", []))
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))
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
      {make_exam_tip(chunk)}
    </div>"""


def layout_bullet_list(chunk, meta):
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))
    if vis:
        inner = f'<div class="two-col"><div class="col-text">{bullets}</div><div class="col-visual">{vis}</div></div>'
    else:
        inner = bullets
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {inner}
      {make_exam_tip(chunk)}
    </div>"""


def layout_visual_explain(chunk, meta):
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      <div class="two-col">
        <div class="col-text">{bullets}</div>
        <div class="col-visual">{vis}</div>
      </div>
      {make_exam_tip(chunk)}
    </div>"""


def layout_visual_full(chunk, meta):
    vis = make_visual(chunk.get("visual", {}))
    bullets = make_bullets(chunk.get("display_bullets", []))
    return f"""
    <div class="content" style="align-items:center;text-align:center">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line" style="margin-left:auto;margin-right:auto"></div>
      <div style="max-width:900px;width:100%">{vis}</div>
      <div style="margin-top:24px;text-align:left;max-width:900px">{bullets}</div>
    </div>"""


def layout_key_takeaway(chunk, meta):
    final_ans = chunk.get("final_answer_display") or chunk.get("final_answer", "")
    bullets = make_bullets(chunk.get("display_bullets", []))
    takeaway = f'<div class="takeaway-box">{final_ans}</div>' if final_ans else ""
    return f"""
    <div class="content" style="justify-content:center">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      {takeaway}
      <div style="margin-top:28px">{bullets}</div>
      {make_exam_tip(chunk)}
    </div>"""


def layout_problem_setup(chunk, meta):
    bullets = make_bullets(chunk.get("display_bullets", []))
    vis = make_visual(chunk.get("visual", {}))
    diff = difficulty_badge(chunk.get("difficulty", ""))
    problem = chunk.get("script_display", "")
    vis_col = f'<div class="col-visual">{vis}</div>' if vis else ""
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')} {diff}</div>
      <div class="accent-line"></div>
      <div class="problem-box">{problem if not bullets else ''}</div>
      {"<div class='two-col'><div class='col-text'>" + bullets + "</div>" + vis_col + "</div>" if vis else bullets}
    </div>"""


def layout_split_comparison(chunk, meta):
    bullets = chunk.get("display_bullets", [])
    mid = len(bullets) // 2 or 1
    left = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets[:mid])
    right = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets[mid:])
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','')}</div>
      <div class="accent-line"></div>
      <div class="two-col" style="gap:40px">
        <div class="compare-col left">
          <div class="compare-title">Correct</div>
          <ul class="bullet-list">{left}</ul>
        </div>
        <div class="compare-col right">
          <div class="compare-title">Incorrect</div>
          <ul class="bullet-list">{right}</ul>
        </div>
      </div>
    </div>"""


def layout_recap_grid(chunk, meta):
    cs = chunk.get("coverage_summary") or {}
    cells = [("Definitions", cs.get("definitions",0)), ("Formulas", cs.get("formulas",0)),
             ("Theorems", cs.get("theorems",0)), ("Properties", cs.get("properties",0)),
             ("Worked Examples", cs.get("worked_examples",0))]
    cells = [(k,v) for k,v in cells if v]
    grid = "".join(
        f'<div class="recap-cell"><div class="recap-count">{v}</div>'
        f'<div class="recap-label">{k}</div></div>' for k,v in cells)
    bullets = make_bullets(chunk.get("display_bullets", []))
    next_mods = chunk.get("next_modules") or []
    next_html = ""
    if next_mods:
        items = " / ".join(next_mods)
        next_html = f'<div style="margin-top:24px;font-size:22px;color:{MUTED}">Next: <span style="color:{ACCENT};font-weight:600">{items}</span></div>'
    return f"""
    <div class="content">
      <div class="slide-title">{chunk.get('slide_title','Recap')}</div>
      <div class="accent-line"></div>
      <div class="two-col">
        <div class="col-text">{bullets}{next_html}</div>
        <div class="col-visual"><div class="recap-grid">{grid}</div></div>
      </div>
    </div>"""


# ── DISPATCH ──

LAYOUT_MAP = {
    "title_hero": layout_title_hero,
    "definition_spotlight": layout_definition_spotlight,
    "formula_showcase": layout_formula_showcase,
    "step_walkthrough": layout_step_walkthrough,
    "bullet_list": layout_bullet_list,
    "visual_explain": layout_visual_explain,
    "visual_full": layout_visual_full,
    "key_takeaway": layout_key_takeaway,
    "problem_setup": layout_problem_setup,
    "split_comparison": layout_split_comparison,
    "recap_grid": layout_recap_grid,
}

TYPE_FALLBACK = {
    "intro": "title_hero",
    "definition": "definition_spotlight",
    "concept_explanation": "bullet_list",
    "formula_derivation": "step_walkthrough",
    "worked_example": "step_walkthrough",
    "recap": "recap_grid",
}


def render_chunk(chunk, meta):
    lc = chunk.get("layout_config") or {}
    name = lc.get("layout", "")
    renderer = LAYOUT_MAP.get(name)
    if not renderer:
        fallback = TYPE_FALLBACK.get(chunk.get("type",""), "bullet_list")
        renderer = LAYOUT_MAP.get(fallback, layout_bullet_list)
    body = renderer(chunk, meta)
    hm = make_header_meta(meta)
    return base_html(body, hm)


# ── SAMPLES ──

SAMPLES = [
    {"label": "title_hero",
     "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/001_intro_0.json",
     "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json"},
    {"label": "definition_spotlight",
     "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/002_definition_0.json",
     "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json"},
    {"label": "formula_showcase",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/006_definition_cont_0_4.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "step_walkthrough",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/025_example_cont_0_1.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "split_comparison",
     "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/012_definition_cont_2_3.json",
     "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json"},
    {"label": "recap_grid",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/032_recap_0.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    # ── 4 NEW SAMPLES ──
    {"label": "bullet_list",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/017_property_0.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "visual_explain",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/004_definition_cont_0_2.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "key_takeaway",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/008_definition_cont_0_6.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "problem_setup",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/024_example_0.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
]


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Design V3 (White + Orange) -- rendering {len(SAMPLES)} sample frames\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for sample in SAMPLES:
            label = sample["label"]
            cp = CHUNKS_DIR / sample["chunk"]
            mp = CHUNKS_DIR / sample["meta"]
            if not cp.exists() or not mp.exists():
                print(f"  SKIP {label}: file missing")
                continue
            with open(cp, "r", encoding="utf-8") as f:
                chunk = json.load(f)
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)

            html = render_chunk(chunk, meta)
            out = OUTPUT_DIR / f"{label}.png"
            await page.set_content(html, wait_until="networkidle")
            await page.wait_for_timeout(800)
            await page.screenshot(path=str(out))
            print(f"  OK  {out}")

        await browser.close()
    print(f"\nDone! Frames at {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
