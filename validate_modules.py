"""
validate_modules.py — production go/no-go validator
Passes all modules for a chapter in one call.
Saves a report per chapter + a batch summary.
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PROJECT_ID     = "project-6565cf16-a3d4-4f6e-935"
LOCATION       = "us-central1"
MASTER_FOLDER  = "master_data"
MODULES_FOLDER = "modules"
PROMPT_PATH    = "prompts/module_validator.txt"
OUTPUT_FOLDER  = "validation_results"
MAX_WORKERS    = 3
DEBUG          = False


# Response schema — array of per-module results
RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "module_id":              {"type": "string"},
            "showstoppers":           {"type": "array", "items": {"type": "string"}},
            "release_recommendation": {"type": "string"},
        },
        "required": ["module_id", "showstoppers", "release_recommendation"],
    },
}


# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()


# ─────────────────────────────────────────
# VALIDATE ENTIRE CHAPTER IN ONE CALL
# ─────────────────────────────────────────

def validate_chapter(chapter_name: str) -> dict | None:

    master_path    = os.path.join(MASTER_FOLDER, f"{chapter_name}.txt")
    chapter_folder = os.path.join(MODULES_FOLDER, chapter_name)
    report_path    = os.path.join(OUTPUT_FOLDER, f"{chapter_name}_validation.json")

    if not os.path.exists(master_path) or not os.path.exists(chapter_folder):
        print(f"[SKIP] {chapter_name} — missing master or modules folder")
        return None

    module_files = sorted([f for f in os.listdir(chapter_folder) if f.endswith(".json")])
    if not module_files:
        print(f"[SKIP] {chapter_name} — no module files")
        return None

    # Resume: skip chapters already fully validated
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("total_modules") == len(module_files):
            print(f"[SKIP] {chapter_name} — already validated")
            return existing

    with open(master_path, "r", encoding="utf-8") as f:
        master_data = f.read()

    modules = []
    for mf in module_files:
        with open(os.path.join(chapter_folder, mf), "r", encoding="utf-8") as f:
            modules.append(json.load(f))

    print(f"Validating: {chapter_name} ({len(modules)} modules) ... ", end="", flush=True)

    prompt = (
        prompt_template
        .replace("{master_data}", master_data)
        .replace("{modules_json}", json.dumps(modules, indent=2, ensure_ascii=False))
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )

        results = json.loads(response.text)

        # Stamp validated_at on each result
        ts = datetime.now(timezone.utc).isoformat()
        for r in results:
            r["validated_at"] = ts

    except Exception as e:
        print(f"ERROR")
        error_path = os.path.join(OUTPUT_FOLDER, f"{chapter_name}_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as ef:
            ef.write(str(e))
        return None

    # Chapter summary
    ready   = sum(1 for r in results if r["release_recommendation"] == "READY_TO_GO")
    blocked = len(results) - ready

    chapter_rec = "DO_NOT_RELEASE" if blocked > 0 else "READY_TO_GO"

    print(chapter_rec)

    report = {
        "chapter_name":           chapter_name,
        "validated_at":           datetime.now(timezone.utc).isoformat(),
        "total_modules":          len(results),
        "ready_to_go":            ready,
        "do_not_release":         blocked,
        "chapter_recommendation": chapter_rec,
        "module_results":         results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ─────────────────────────────────────────
# BATCH SUMMARY
# ─────────────────────────────────────────

def save_batch_summary(reports: list[dict]):

    total   = sum(r["total_modules"] for r in reports)
    ready   = sum(r["ready_to_go"]   for r in reports)
    blocked = sum(r["do_not_release"] for r in reports)

    batch_rec = "DO_NOT_RELEASE" if blocked > 0 else "READY_TO_GO"

    flagged_modules = [
        {
            "chapter":        r["chapter_name"],
            "module_id":      m["module_id"],
            "showstoppers":   m.get("showstoppers", []),
        }
        for r in reports
        for m in r["module_results"]
        if m["release_recommendation"] != "READY_TO_GO"
    ]

    summary = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "batch_recommendation": batch_rec,
        "total_chapters":     len(reports),
        "total_modules":      total,
        "ready_to_go":        ready,
        "do_not_release":     blocked,
        "flagged_modules":    flagged_modules,
        "chapter_summaries":  [
            {
                "chapter":    r["chapter_name"],
                "recommendation": r["chapter_recommendation"],
                "modules":    r["total_modules"],
                "ready":      r["ready_to_go"],
                "blocked":    r["do_not_release"],
            }
            for r in reports
        ],
    }

    summary_path = os.path.join(OUTPUT_FOLDER, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nBatch summary → {summary_path}")
    print(f"{batch_rec} | {len(reports)} chapters | {total} modules (✓{ready} ✗{blocked})")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

async def main():

    chapter_names = sorted([
        d for d in os.listdir(MODULES_FOLDER)
        if os.path.isdir(os.path.join(MODULES_FOLDER, d))
    ])

    if not chapter_names:
        print("No chapter folders found in modules/")
        return

    if DEBUG:
        chapter_names = chapter_names[:1]
        print(f"[DEBUG] Processing only: {chapter_names}")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [loop.run_in_executor(executor, validate_chapter, ch) for ch in chapter_names]
        raw_reports = await asyncio.gather(*tasks)

    reports = [r for r in raw_reports if r is not None]

    if reports:
        save_batch_summary(reports)
    else:
        print("No new validations were run.")


if __name__ == "__main__":
    asyncio.run(main())