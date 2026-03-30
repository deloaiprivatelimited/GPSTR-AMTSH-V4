"""
validate_modules.py — simple content quality validator
Asks Gemini: "is this module good enough for production?"
Saves one JSON result per module + a batch summary.
"""

import os
import re
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
MAX_WORKERS    = 2
DEBUG          = False


# Minimal schema — flat, Vertex-safe
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "coverage_status":       {"type": "string"},
        "subject_accuracy":      {"type": "string"},
        "concept_integrity":     {"type": "string"},
        "diagram_integrity":     {"type": "string"},
        "structural_compliance": {"type": "string"},
        "critical_issues":       {"type": "array", "items": {"type": "string"}},
        "major_issues":          {"type": "array", "items": {"type": "string"}},
        "minor_issues":          {"type": "array", "items": {"type": "string"}},
        "release_recommendation":{"type": "string"},
    },
    "required": [
        "coverage_status",
        "subject_accuracy",
        "concept_integrity",
        "diagram_integrity",
        "structural_compliance",
        "critical_issues",
        "major_issues",
        "minor_issues",
        "release_recommendation",
    ],
}


# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-flash")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()


# ─────────────────────────────────────────
# HELPER: CHUNK MASTER DATA
# ─────────────────────────────────────────
def extract_concept_data(master_data_full: str, target_concept_id: str) -> str:
    blocks = re.split(r'━{5,}', master_data_full)

    metadata_block = ""
    concept_block = ""

    for block in blocks:

        if "[ಅಧ್ಯಾಯ_ಮೆಟಾಡೇಟಾ]" in block:
            metadata_block = block.strip()

        if (
            f'"concept_id": "{target_concept_id}"' in block
            or f"concept_id: {target_concept_id}" in block
        ):
            concept_block = block.strip()

    if not concept_block:
        print(f"⚠️ Concept ID {target_concept_id} not found in master data. Using full master data as fallback.")
        return master_data_full

    return f"{metadata_block}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{concept_block}"
# ─────────────────────────────────────────
# VALIDATE ONE MODULE
# ─────────────────────────────────────────
def validate_module(chunked_master_data: str, module: dict) -> dict:

    prompt = (
        prompt_template
        .replace("{master_data}", chunked_master_data)
        .replace("{module_json}", json.dumps(module, indent=2, ensure_ascii=False))
    )

    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )

    result = json.loads(response.text)

    result["module_id"] = module.get("module_id", "unknown")
    result["validated_at"] = datetime.now(timezone.utc).isoformat()

    return result


# ─────────────────────────────────────────
# VALIDATE ONE CHAPTER
# ─────────────────────────────────────────
def validate_chapter(chapter_name: str) -> dict | None:

    master_path = os.path.join(MASTER_FOLDER, f"{chapter_name}.txt")
    chapter_folder = os.path.join(MODULES_FOLDER, chapter_name)

    report_path = os.path.join(OUTPUT_FOLDER, f"{chapter_name}_validation.json")

    existing_results = []
    validated_ids = set()

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            existing_report = json.load(f)

        existing_results = existing_report.get("module_results", [])
        validated_ids = {m["module_id"] for m in existing_results}
        print(f"[RESUME] {chapter_name} — {len(validated_ids)} modules already validated")

    if not os.path.exists(master_path) or not os.path.exists(chapter_folder):
        print(f"[SKIP] {chapter_name} — missing master or modules folder")
        return None

    module_files = sorted([f for f in os.listdir(chapter_folder) if f.endswith(".json")])

    if not module_files:
        print(f"[SKIP] {chapter_name} — no module files")
        return None

    # Load the full master data once per chapter
    with open(master_path, "r", encoding="utf-8") as f:
        master_data_full = f.read()

    print(f"\nValidating: {chapter_name} ({len(module_files)} modules)")

    results = existing_results.copy()

    for mf in module_files:
        module_path = os.path.join(chapter_folder, mf)

        with open(module_path, "r", encoding="utf-8") as f:
            module = json.load(f)

        module_id = module.get("module_id", mf.replace(".json", ""))
        
        if module_id in validated_ids:
            continue
            
        print(f"  {module_id} ... ", end="", flush=True)

        try:
            # OPTION A APPLIED: Extract only the relevant chunk for this specific module
            chunked_master_data = extract_concept_data(master_data_full, module_id)
            
            result = validate_module(chunked_master_data, module)
            results.append(result)
            print(result["release_recommendation"])

        except Exception as e:
            print(f"ERROR")
            error_path = os.path.join(OUTPUT_FOLDER, f"{module_id}_ERROR.txt")
            with open(error_path, "w", encoding="utf-8") as ef:
                ef.write(str(e))

    if not results:
        return None

    # Chapter summary
    ready = sum(1 for r in results if r["release_recommendation"] == "READY_TO_GO")
    review = sum(1 for r in results if r["release_recommendation"] == "NEEDS_REVIEW")
    blocked = sum(1 for r in results if r["release_recommendation"] == "DO_NOT_RELEASE")

    chapter_rec = (
        "DO_NOT_RELEASE" if blocked > 0 else
        "NEEDS_REVIEW" if review > 0 else
        "READY_TO_GO"
    )

    report = {
        "chapter_name": chapter_name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "total_modules": len(results),
        "ready_to_go": ready,
        "needs_review": review,
        "do_not_release": blocked,
        "chapter_recommendation": chapter_rec,
        "module_results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved → {report_path} [{chapter_rec}]")
    return report


# ─────────────────────────────────────────
# BATCH SUMMARY
# ─────────────────────────────────────────
def save_batch_summary(reports: list[dict]):

    total = sum(r["total_modules"] for r in reports)
    ready = sum(r["ready_to_go"] for r in reports)
    review = sum(r["needs_review"] for r in reports)
    blocked = sum(r["do_not_release"] for r in reports)

    batch_rec = (
        "DO_NOT_RELEASE" if blocked > 0 else
        "NEEDS_REVIEW" if review > 0 else
        "READY_TO_GO"
    )

    flagged_modules = [
        {
            "chapter": r["chapter_name"],
            "module_id": m["module_id"],
            "recommendation": m["release_recommendation"],
            "critical_issues": m.get("critical_issues", []),
            "major_issues": m.get("major_issues", []),
        }
        for r in reports for m in r["module_results"]
        if m["release_recommendation"] != "READY_TO_GO"
    ]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_recommendation": batch_rec,
        "total_chapters": len(reports),
        "total_modules": total,
        "ready_to_go": ready,
        "needs_review": review,
        "do_not_release": blocked,
        "flagged_modules": flagged_modules,
        "chapter_summaries": [
            {
                "chapter": r["chapter_name"],
                "recommendation": r["chapter_recommendation"],
                "modules": r["total_modules"],
                "ready": r["ready_to_go"],
                "review": r["needs_review"],
                "blocked": r["do_not_release"],
            }
            for r in reports
        ],
    }

    summary_path = os.path.join(OUTPUT_FOLDER, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nBatch summary saved →", summary_path)
    print(
        f"{batch_rec} | {len(reports)} chapters | {total} modules "
        f"(✓{ready} ⚠{review} ✗{blocked})"
    )


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
        tasks = [
            loop.run_in_executor(executor, validate_chapter, ch)
            for ch in chapter_names
        ]
        raw_reports = await asyncio.gather(*tasks)

    reports = [r for r in raw_reports if r is not None]

    if reports:
        save_batch_summary(reports)
    else:
        print("No new validations were run.")


if __name__ == "__main__":
    asyncio.run(main())