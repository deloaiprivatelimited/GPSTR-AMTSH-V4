"""
design_compare.py
------------------
Renders the SAME 4 chunks in 4 DIFFERENT design themes for comparison.
Output: claude_works/design_compare/<theme>/<layout>.png
"""

import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

WIDTH, HEIGHT = 1920, 1080
CHUNKS_DIR = Path("claude_works/chunks_structured")
OUTPUT_DIR = Path("claude_works/design_compare")

# ══════════════════════════════════════
# 4 CHUNKS (same across all themes)
# ══════════════════════════════════════

SAMPLES = [
    {"label": "01_title_hero",
     "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/001_intro_0.json",
     "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json"},
    {"label": "02_formula",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/006_definition_cont_0_4.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "03_step_walkthrough",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/025_example_cont_0_1.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
    {"label": "04_visual_explain",
     "chunk": "ALG-AP-4/ALG-AP-4_concept_1/004_definition_cont_0_2.json",
     "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json"},
]


# ══════════════════════════════════════
# 4 THEME DEFINITIONS
# ══════════════════════════════════════

THEMES = {

    # ── A: White + Orange (current v3) ──
    "A_white_orange": {
        "ACCENT": "#E8651A", "ACCENT2": "#F28C28",
        "BG": "#FFFFFF", "TEXT": "#1A1A1A", "SUBTEXT": "#4A4A4A",
        "MUTED": "#888888", "LIGHT_BG": "#FFF7F0", "CARD_BG": "#FEF0E5",
        "BORDER": "#F0D4BC",
        "HEADER_BG": "#E8651A", "HEADER_TEXT": "#FFFFFF",
        "EXAM_BG": "#FFFDE6", "EXAM_BORDER": "#E6D96B", "EXAM_LABEL": "#B8960C",
        "COMPARE_RIGHT_BG": "#FFF5F5", "COMPARE_RIGHT_BORDER": "#F5C6C6",
        "COMPARE_RIGHT_TITLE": "#CC4444",
    },

    # ── B: Dark Navy + Gold ──
    "B_navy_gold": {
        "ACCENT": "#D4A843", "ACCENT2": "#C9952E",
        "BG": "#0D1B2A", "TEXT": "#F0F0F0", "SUBTEXT": "#B8C4D0",
        "MUTED": "#6B7D8E", "LIGHT_BG": "rgba(212,168,67,0.08)", "CARD_BG": "rgba(212,168,67,0.12)",
        "BORDER": "rgba(212,168,67,0.25)",
        "HEADER_BG": "#142640", "HEADER_TEXT": "#D4A843",
        "EXAM_BG": "rgba(212,168,67,0.08)", "EXAM_BORDER": "rgba(212,168,67,0.3)", "EXAM_LABEL": "#D4A843",
        "COMPARE_RIGHT_BG": "rgba(255,100,100,0.06)", "COMPARE_RIGHT_BORDER": "rgba(255,100,100,0.2)",
        "COMPARE_RIGHT_TITLE": "#FF9999",
    },

    # ── C: Light Teal + Slate ──
    "C_teal_slate": {
        "ACCENT": "#0A8F8F", "ACCENT2": "#0DBDBD",
        "BG": "#F5F9FA", "TEXT": "#1C2D36", "SUBTEXT": "#3D5A6E",
        "MUTED": "#7A95A5", "LIGHT_BG": "#EAF4F4", "CARD_BG": "#DFF0F0",
        "BORDER": "#B5D8D8",
        "HEADER_BG": "#0A8F8F", "HEADER_TEXT": "#FFFFFF",
        "EXAM_BG": "#F0FFF0", "EXAM_BORDER": "#8BC48B", "EXAM_LABEL": "#2E7D32",
        "COMPARE_RIGHT_BG": "#FFF0F0", "COMPARE_RIGHT_BORDER": "#E0B0B0",
        "COMPARE_RIGHT_TITLE": "#B44444",
    },

    # ── D: Warm Cream + Deep Red ──
    "D_cream_red": {
        "ACCENT": "#B82E2E", "ACCENT2": "#D44444",
        "BG": "#FBF8F4", "TEXT": "#2A1F1F", "SUBTEXT": "#5A4A4A",
        "MUTED": "#9A8A8A", "LIGHT_BG": "#F5EFEA", "CARD_BG": "#FAEDEB",
        "BORDER": "#E0CBC5",
        "HEADER_BG": "#B82E2E", "HEADER_TEXT": "#FFFFFF",
        "EXAM_BG": "#FFF8E6", "EXAM_BORDER": "#D4B85C", "EXAM_LABEL": "#8B7520",
        "COMPARE_RIGHT_BG": "#EFF5FF", "COMPARE_RIGHT_BORDER": "#B0C8E8",
        "COMPARE_RIGHT_TITLE": "#3366AA",
    },
}


# ══════════════════════════════════════
# STYLE TEMPLATE (tokens injected per theme)
# ══════════════════════════════════════

def build_styles(t):
    return f"""
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Kannada:wght@400;500;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {WIDTH}px; height: {HEIGHT}px;
    background: {t['BG']};
    font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
    color: {t['TEXT']};
    display: flex; flex-direction: column;
    overflow: hidden;
  }}
  .header {{
    width: 100%; height: 56px;
    display: flex; align-items: center;
    padding: 0 64px;
    background: {t['HEADER_BG']};
    flex-shrink: 0;
  }}
  .brand {{
    font-size: 15px; font-weight: 700;
    letter-spacing: 3px; color: {t['HEADER_TEXT']};
    text-transform: uppercase;
  }}
  .header-meta {{
    margin-left: auto;
    font-size: 13px; font-weight: 500;
    color: {t['HEADER_TEXT']}; opacity: 0.8;
  }}
  .content {{
    flex: 1; padding: 48px 72px;
    display: flex; flex-direction: column;
    justify-content: center; overflow: hidden;
  }}
  .slide-title {{
    font-size: 46px; font-weight: 800;
    line-height: 1.25; margin-bottom: 10px;
  }}
  .accent-line {{
    width: 80px; height: 4px;
    background: {t['ACCENT']};
    margin-bottom: 36px; border-radius: 2px;
  }}
  .two-col {{ display: flex; gap: 56px; align-items: flex-start; flex-grow: 1; }}
  .col-text {{ flex: 1; }}
  .col-visual {{ flex: 0 0 500px; }}

  .bullet-list {{ list-style: none; }}
  .bullet-list li {{
    font-size: 30px; line-height: 1.55;
    margin-bottom: 20px; display: flex;
    align-items: flex-start; color: {t['SUBTEXT']};
  }}
  .bullet-node {{
    width: 6px; height: 24px;
    background: {t['ACCENT']}; margin-top: 10px;
    margin-right: 22px; flex-shrink: 0; border-radius: 1px;
  }}

  .step-card {{
    border-radius: 10px; padding: 18px 24px;
    margin-bottom: 14px;
    border-left: 4px solid {t['ACCENT2']};
    font-size: 28px; line-height: 1.5;
    color: {t['SUBTEXT']};
    background: {t['LIGHT_BG']};
  }}
  .step-num {{
    font-size: 18px; font-weight: 700;
    color: {t['ACCENT']}; margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  .step-justification {{
    font-size: 20px; color: {t['MUTED']};
    margin-top: 4px; font-style: italic;
  }}

  .formula-box {{
    background: {t['LIGHT_BG']};
    border: 2px solid {t['BORDER']};
    border-radius: 14px; padding: 28px 36px;
    text-align: center; font-size: 42px;
    margin-bottom: 24px;
  }}
  .formula-label {{
    font-size: 18px; color: {t['MUTED']};
    margin-bottom: 12px; text-transform: uppercase;
    letter-spacing: 2px; font-weight: 600;
  }}

  table {{
    border-collapse: collapse; width: 100%;
    font-size: 24px; table-layout: fixed;
  }}
  th {{
    background: {t['ACCENT']};
    color: {t['HEADER_TEXT']}; padding: 14px 20px;
    text-align: left; font-weight: 600;
    word-wrap: break-word; overflow-wrap: break-word;
  }}
  td {{
    padding: 12px 20px; color: {t['SUBTEXT']};
    border-bottom: 1px solid {t['BORDER']};
    word-wrap: break-word; overflow-wrap: break-word; vertical-align: top;
  }}

  .diagram-placeholder {{
    border: 2px dashed {t['BORDER']};
    border-radius: 12px; padding: 40px;
    text-align: center; color: {t['MUTED']};
    font-size: 20px; background: {t['LIGHT_BG']};
  }}

  .problem-box {{
    font-size: 28px; color: {t['SUBTEXT']};
    background: {t['LIGHT_BG']};
    padding: 18px 24px; border-radius: 10px;
    border-left: 4px solid {t['ACCENT']};
    margin-bottom: 24px; line-height: 1.6;
  }}

  .answer-box {{
    margin-top: 18px; padding: 18px 24px;
    background: {t['CARD_BG']};
    border-radius: 10px; font-size: 32px;
    font-weight: 700; color: {t['ACCENT']};
    border: 2px solid {t['ACCENT2']};
  }}

  .badge {{
    font-size: 18px; padding: 4px 14px;
    border-radius: 20px; margin-left: 14px;
    vertical-align: middle; display: inline-block; font-weight: 600;
  }}
  .badge-basic {{ color: #2E8B57; border: 1.5px solid #2E8B57; }}
  .badge-intermediate {{ color: {t['ACCENT']}; border: 1.5px solid {t['ACCENT']}; }}
  .badge-advanced {{ color: #CC4444; border: 1.5px solid #CC4444; }}

  .exam-tip {{
    background: {t['EXAM_BG']};
    border: 1px solid {t['EXAM_BORDER']};
    border-radius: 10px; padding: 18px 24px;
    font-size: 24px; margin-top: 20px;
  }}
  .exam-tip-label {{
    font-size: 16px; font-weight: 700;
    color: {t['EXAM_LABEL']}; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 6px;
  }}

  .takeaway-box {{
    background: {t['CARD_BG']};
    border: 2px solid {t['ACCENT2']};
    border-radius: 14px; padding: 34px 40px;
    font-size: 34px; line-height: 1.6; text-align: center;
  }}

  .recap-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 8px; }}
  .recap-cell {{
    background: {t['LIGHT_BG']}; border: 2px solid {t['BORDER']};
    border-radius: 12px; padding: 22px; text-align: center;
  }}
  .recap-count {{ font-size: 50px; font-weight: 800; color: {t['ACCENT']}; line-height: 1; }}
  .recap-label {{ font-size: 20px; color: {t['SUBTEXT']}; margin-top: 6px; }}

  .compare-col {{ flex: 1; padding: 28px; border-radius: 14px; }}
  .compare-col.left {{
    background: {t['LIGHT_BG']}; border: 2px solid {t['BORDER']};
  }}
  .compare-col.right {{
    background: {t['COMPARE_RIGHT_BG']}; border: 2px solid {t['COMPARE_RIGHT_BORDER']};
  }}
  .compare-title {{ font-size: 26px; font-weight: 700; margin-bottom: 16px; }}
  .compare-col.left .compare-title {{ color: {t['ACCENT']}; }}
  .compare-col.right .compare-title {{ color: {t['COMPARE_RIGHT_TITLE']}; }}
"""


# ══════════════════════════════════════
# HTML WRAPPER
# ══════════════════════════════════════

def base_html(body, header_meta, styles):
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[
    {{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}
  ]}});"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Kannada:wght@400;500;700&display=swap" rel="stylesheet">
<style>{styles}</style>
</head><body>
  <div class="header">
    <div class="brand">Srinivas IAS Academy</div>
    <div class="header-meta">{header_meta}</div>
  </div>
  {body}
</body></html>"""


# ══════════════════════════════════════
# HELPERS (same for all themes)
# ══════════════════════════════════════

def hm(meta):
    return f"{meta.get('class','')}ನೇ ತರಗತಿ &nbsp;|&nbsp; ಅಧ್ಯಾಯ {meta.get('chapter','')}"

def bul(bullets):
    if not bullets: return ""
    items = "".join(f"<li><span class='bullet-node'></span>{b}</li>" for b in bullets)
    return f'<ul class="bullet-list">{items}</ul>'

def vis(visual, t):
    if not visual or visual.get("type") == "none": return ""
    vt = visual.get("type","")
    if vt == "formula_box":
        latex = visual.get("latex","")
        desc = visual.get("description","")
        lbl = f'<div class="formula-label">{desc}</div>' if desc else ""
        return f'<div class="formula-box">{lbl}$${latex}$$</div>' if latex else ""
    if vt == "diagram":
        return (f'<div class="diagram-placeholder">[{visual.get("description","diagram")}]'
                f'<br><span style="font-size:15px">{visual.get("mathematical_significance","")}</span></div>')
    if vt == "table":
        headers = visual.get("headers") or []
        rows = visual.get("rows") or []
        th = "".join(f"<th>{h}</th>" for h in headers)
        tr = "".join("<tr>"+"".join(f"<td>{c}</td>" for c in row)+"</tr>" for row in rows)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"
    return ""

def steps_html(steps):
    if not steps: return ""
    c = ""
    for s in steps:
        j = s.get("justification","")
        jh = f'<div class="step-justification">{j}</div>' if j else ""
        c += f'<div class="step-card"><div class="step-num">Step {s.get("step","")}</div><div>{s.get("action_display","")}</div>{jh}</div>'
    return c


# ══════════════════════════════════════
# LAYOUT RENDERERS
# ══════════════════════════════════════

def render_chunk_html(chunk, meta, t):
    layout = (chunk.get("layout_config") or {}).get("layout", "")
    ctype = chunk.get("type", "")
    fallbacks = {"intro":"title_hero","definition":"definition_spotlight",
                 "concept_explanation":"bullet_list","worked_example":"step_walkthrough","recap":"recap_grid"}
    if not layout:
        layout = fallbacks.get(ctype, "bullet_list")

    bullets = bul(chunk.get("display_bullets", []))
    visual = vis(chunk.get("visual", {}), t)
    title = chunk.get("slide_title", "")

    if layout == "title_hero":
        domain = meta.get("domain","")
        cls = meta.get("class","")
        sub = f"<div style='font-size:22px;color:{t['ACCENT']};letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;font-weight:600'>{domain} - {cls}ನೇ ತರಗತಿ</div>" if domain else ""
        vc = f'<div class="col-visual">{visual}</div>' if visual else ""
        inner = f'<div class="two-col"><div class="col-text">{bullets}</div>{vc}</div>' if visual else bullets
        body = f'<div class="content" style="justify-content:center">{sub}<div class="slide-title" style="font-size:56px">{title}</div><div class="accent-line"></div>{inner}</div>'

    elif layout == "formula_showcase":
        if not visual:
            v = chunk.get("visual",{})
            if v.get("latex"):
                visual = f'<div class="formula-box" style="font-size:46px">$${v["latex"]}$$</div>'
        body = f'<div class="content"><div class="slide-title">{title}</div><div class="accent-line"></div><div class="two-col"><div class="col-text">{bullets}</div><div class="col-visual">{visual}</div></div></div>'

    elif layout == "step_walkthrough":
        st = steps_html(chunk.get("steps",[]))
        text = st if st else bullets
        fa = chunk.get("final_answer_display") or chunk.get("final_answer","")
        fh = f'<div class="answer-box">{fa}</div>' if fa else ""
        if visual:
            inner = f'<div class="two-col"><div class="col-text" style="overflow-y:auto;max-height:780px">{text}{fh}</div><div class="col-visual">{visual}</div></div>'
        else:
            inner = f'<div style="overflow-y:auto;max-height:780px">{text}{fh}</div>'
        body = f'<div class="content"><div class="slide-title">{title}</div><div class="accent-line"></div>{inner}</div>'

    elif layout == "visual_explain":
        body = f'<div class="content"><div class="slide-title">{title}</div><div class="accent-line"></div><div class="two-col"><div class="col-text">{bullets}</div><div class="col-visual">{visual}</div></div></div>'

    else:  # bullet_list / definition_spotlight / fallback
        if visual:
            inner = f'<div class="two-col"><div class="col-text">{bullets}</div><div class="col-visual">{visual}</div></div>'
        else:
            inner = bullets
        body = f'<div class="content"><div class="slide-title">{title}</div><div class="accent-line"></div>{inner}</div>'

    styles = build_styles(t)
    return base_html(body, hm(meta), styles)


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════

async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(THEMES) * len(SAMPLES)
    print(f"Rendering {len(SAMPLES)} chunks x {len(THEMES)} themes = {total} frames\n")

    # Load chunk data once
    loaded = []
    for s in SAMPLES:
        cp = CHUNKS_DIR / s["chunk"]
        mp = CHUNKS_DIR / s["meta"]
        with open(cp, "r", encoding="utf-8") as f: chunk = json.load(f)
        with open(mp, "r", encoding="utf-8") as f: meta = json.load(f)
        loaded.append({"label": s["label"], "chunk": chunk, "meta": meta})

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for theme_name, tokens in THEMES.items():
            theme_dir = OUTPUT_DIR / theme_name
            theme_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Theme: {theme_name}")

            for item in loaded:
                html = render_chunk_html(item["chunk"], item["meta"], tokens)
                out = theme_dir / f"{item['label']}.png"
                await page.set_content(html, wait_until="networkidle")
                await page.wait_for_timeout(700)
                await page.screenshot(path=str(out))
                print(f"    OK  {item['label']}")

        await browser.close()

    print(f"\nDone! Compare at {OUTPUT_DIR}/")
    for t in THEMES:
        print(f"  {OUTPUT_DIR}/{t}/")


if __name__ == "__main__":
    asyncio.run(main())
