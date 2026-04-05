"""
generate_sample_frames.py
--------------------------
Picks one chunk per layout type from chunks_structured/ and renders
sample frame PNGs using generate_video_v2's layout engine.

Output: claude_works/sample_frames/<layout_name>.png
"""

import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Import the rendering engine from generate_video_v2
from generate_video_v2 import (
    render_chunk_html, LAYOUT_MAP, WIDTH, HEIGHT
)

CHUNKS_DIR = Path("claude_works/chunks_structured")
OUTPUT_DIR = Path("claude_works/sample_frames")

# Hand-picked samples: (chunk_path, meta_path) covering different layouts
SAMPLES = [
    # title_hero (intro slide)
    {
        "label": "title_hero",
        "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/001_intro_0.json",
        "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json",
    },
    # definition_spotlight
    {
        "label": "definition_spotlight",
        "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/002_definition_0.json",
        "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json",
    },
    # formula_showcase
    {
        "label": "formula_showcase",
        "chunk": "ALG-AP-4/ALG-AP-4_concept_1/006_definition_cont_0_4.json",
        "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json",
    },
    # step_walkthrough
    {
        "label": "step_walkthrough",
        "chunk": "ALG-AP-4/ALG-AP-4_concept_1/025_example_cont_0_1.json",
        "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json",
    },
    # split_comparison (with table visual)
    {
        "label": "split_comparison_with_table",
        "chunk": "ALG-EQ1-1/ALG-EQ1-1_concept_1/012_definition_cont_2_3.json",
        "meta":  "ALG-EQ1-1/ALG-EQ1-1_concept_1/_meta.json",
    },
    # recap_grid
    {
        "label": "recap_grid",
        "chunk": "ALG-AP-4/ALG-AP-4_concept_1/032_recap_0.json",
        "meta":  "ALG-AP-4/ALG-AP-4_concept_1/_meta.json",
    },
]


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(SAMPLES)} sample frames...\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        for sample in SAMPLES:
            label = sample["label"]
            chunk_path = CHUNKS_DIR / sample["chunk"]
            meta_path  = CHUNKS_DIR / sample["meta"]

            if not chunk_path.exists() or not meta_path.exists():
                print(f"  SKIP {label}: file not found")
                continue

            with open(chunk_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            html = render_chunk_html(chunk_data, meta)
            out_path = OUTPUT_DIR / f"{label}.png"

            await page.set_content(html, wait_until="networkidle")
            await page.wait_for_timeout(800)  # KaTeX render time
            await page.screenshot(path=str(out_path))

            print(f"  OK  {out_path}")

        await browser.close()

    print(f"\nDone! {len(SAMPLES)} frames saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
