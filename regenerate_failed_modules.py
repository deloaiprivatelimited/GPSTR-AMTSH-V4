"""
regenerate_failed_modules.py
────────────────────────────
Reads validation_results/ → finds modules that are NOT READY_TO_GO
→ extracts ONLY that concept's block from master data
→ sends concept block + validation report (issues) to Gemini
→ Gemini fixes exactly what the validator flagged
→ saves ONE module JSON

SET DRY_RUN = True first to see what will be regenerated without touching anything.
"""

import os
import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
from pydantic import BaseModel, Field

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PROJECT_ID        = "project-6565cf16-a3d4-4f6e-935"
LOCATION          = "us-central1"
MASTER_FOLDER     = "master_data"
MODULES_FOLDER    = "modules"
VALIDATION_FOLDER = "validation_results"
PROMPT_PATH       = "prompts/module_splitter.txt"
MAX_WORKERS       = 4

DRY_RUN = False    # ← SET TO False TO ACTUALLY REGENERATE
DEBUG   = False   # ← SET TO True TO PROCESS ONLY FIRST FAILED MODULE

REGEN_STATUSES = {"DO_NOT_RELEASE"}


# ─────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────

class Definition(BaseModel):
    term: str
    definition: str

class Formula(BaseModel):
    formula_name: str
    latex: str
    condition: str
    exception: Optional[str] = None
    derived_from: Optional[str] = None
    used_in: str

class Theorem(BaseModel):
    theorem_name: str
    statement: str
    proof_sketch: Optional[str] = None
    converse: Optional[str] = None
    special_cases: List[str] = Field(default_factory=list)
    common_error: str

class ExampleStep(BaseModel):
    step: int
    action: str
    justification: str

class WorkedExample(BaseModel):
    example_id: str
    difficulty: str
    problem: str
    steps: List[ExampleStep]
    result: str
    common_trap: Optional[str] = None

class Theory(BaseModel):
    concept_summary: str
    definitions: List[Definition] = Field(default_factory=list)
    formulas: List[Formula] = Field(default_factory=list)
    theorems: List[Theorem] = Field(default_factory=list)
    properties: List[str] = Field(default_factory=list)
    visual_aids: List[dict] = Field(default_factory=list)

class ExamIntelligence(BaseModel):
    gpstr_weightage: str
    mcq_note: str
    two_mark_note: str
    five_mark_note: str
    common_mistakes: List[str] = Field(default_factory=list)
    boundary_conditions: List[str] = Field(default_factory=list)
    connects_to: List[str] = Field(default_factory=list)

class Module(BaseModel):
    module_id: str
    module_title: str
    class_name: str = Field(alias="class")
    chapter: str
    chapter_title: str
    domain: str
    prerequisites: List[str] = Field(default_factory=list)
    theory: Theory
    worked_examples: List[WorkedExample] = Field(default_factory=list)
    exam_intelligence: ExamIntelligence
    model_config = {"populate_by_name": True}

class ModuleList(BaseModel):
    modules: List[Module]


# ─────────────────────────────────────────
# VERTEX SCHEMA CLEANER
# ─────────────────────────────────────────

def get_vertex_safe_schema(model_class):
    full_schema = model_class.model_json_schema()
    definitions = full_schema.get("$defs", {})

    def resolve_and_clean(node):
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_key = node["$ref"].split("/")[-1]
            return resolve_and_clean(definitions[ref_key])
        if "anyOf" in node:
            real_options = [opt for opt in node["anyOf"] if opt.get("type") != "null"]
            if real_options:
                return resolve_and_clean(real_options[0])
        if "properties" in node:
            node["properties"] = {k: resolve_and_clean(v) for k, v in node["properties"].items()}
        if "items" in node:
            node["items"] = resolve_and_clean(node["items"])
        node.pop("title", None)
        node.pop("default", None)
        node.pop("additionalProperties", None)
        node.pop("$defs", None)
        return node

    return resolve_and_clean(full_schema)


# ─────────────────────────────────────────
# EXTRACT ONE CONCEPT BLOCK FROM MASTER DATA
# ─────────────────────────────────────────

def extract_concept_block(master_data_full: str, concept_id: str) -> tuple[str, bool]:
    blocks = re.split(r'━{5,}', master_data_full)

    metadata_block = ""
    concept_block  = ""

    for block in blocks:
        if "[ಅಧ್ಯಾಯ_ಮೆಟಾಡೇಟಾ]" in block:
            metadata_block = block.strip()
        if (
            f'"concept_id": "{concept_id}"' in block
            or f"concept_id: {concept_id}" in block
        ):
            concept_block = block.strip()

    if not concept_block:
        return master_data_full, False

    text = (
        f"{metadata_block}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{concept_block}"
    )
    return text, True


# ─────────────────────────────────────────
# FORMAT VALIDATION REPORT AS REVIEWER NOTES
# ─────────────────────────────────────────

def format_validation_feedback(val_result: dict) -> str:
    """
    Turns the validator's JSON result into a clear reviewer-notes block
    that Gemini can read and act on.
    """
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "QA REVIEWER FEEDBACK (from previous generation)",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Overall verdict      : {val_result.get('release_recommendation', '?')}",
        f"Coverage             : {val_result.get('coverage_status', '?')}",
        f"Subject accuracy     : {val_result.get('subject_accuracy', '?')}",
        f"Concept integrity    : {val_result.get('concept_integrity', '?')}",
        f"Diagram integrity    : {val_result.get('diagram_integrity', '?')}",
        f"Structural compliance: {val_result.get('structural_compliance', '?')}",
        "",
    ]

    critical = val_result.get("critical_issues", [])
    if critical:
        lines.append("🔴 CRITICAL ISSUES — must fix:")
        for issue in critical:
            lines.append(f"  • {issue}")
        lines.append("")

    major = val_result.get("major_issues", [])
    if major:
        lines.append("🟠 MAJOR ISSUES — fix before release:")
        for issue in major:
            lines.append(f"  • {issue}")
        lines.append("")

    minor = val_result.get("minor_issues", [])
    if minor:
        lines.append("🟡 MINOR ISSUES — improve if possible:")
        for issue in minor:
            lines.append(f"  • {issue}")
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "YOUR TASK: Regenerate this module fixing ALL issues listed above.",
        "Do not change anything that was not flagged.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────
# COLLECT FAILED MODULES + THEIR VAL RESULTS
# ─────────────────────────────────────────

def collect_failed_modules() -> dict[str, list[dict]]:
    """
    Returns {
      chapter_name: [
        { "module_id": "...", "val_result": { ...validator JSON... } },
        ...
      ]
    }
    """
    failed: dict[str, list[dict]] = {}

    abs_val = os.path.abspath(VALIDATION_FOLDER)
    print(f"\n[SCAN] Validation folder: {abs_val}")

    if not os.path.exists(VALIDATION_FOLDER):
        print(f"  ❌ Folder not found. Fix VALIDATION_FOLDER in CONFIG.")
        return failed

    val_files = [f for f in os.listdir(VALIDATION_FOLDER) if f.endswith("_validation.json")]
    print(f"  Found {len(val_files)} validation file(s): {val_files or 'NONE'}\n")

    for fname in val_files:
        chapter_name = fname.replace("_validation.json", "")
        with open(os.path.join(VALIDATION_FOLDER, fname), "r", encoding="utf-8") as f:
            report = json.load(f)

        all_results = report.get("module_results", [])
        bad_entries = []

        print(f"  [{chapter_name}]  {len(all_results)} module(s):")
        for m in all_results:
            status = m.get("release_recommendation", "UNKNOWN")
            marker = "✗" if status in REGEN_STATUSES else "✓"
            print(f"    {marker}  {m.get('module_id','?'):35s}  {status}")
            if status in REGEN_STATUSES:
                bad_entries.append({
                    "module_id":  m["module_id"],
                    "val_result": m,           # full validator JSON for this module
                })

        if bad_entries:
            failed[chapter_name] = bad_entries

    return failed


# ─────────────────────────────────────────
# CLEAR OLD VALIDATION FOR ONE MODULE
# ─────────────────────────────────────────

def clear_validation_result(chapter_name: str, module_id: str):
    report_path = os.path.join(VALIDATION_FOLDER, f"{chapter_name}_validation.json")
    if not os.path.exists(report_path):
        return
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["module_results"] = [
        m for m in report.get("module_results", [])
        if m.get("module_id") != module_id
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"    🗑  cleared old validation for {module_id}")


# ─────────────────────────────────────────
# REGENERATE ONE MODULE — one API call
# ─────────────────────────────────────────

def regenerate_one_module(
    chapter_name: str,
    module_id: str,
    val_result: dict,
    master_data_full: str,
    splitter_prompt: str,
    model: GenerativeModel,
):
    concept_text, found = extract_concept_block(master_data_full, module_id)
    if not found:
        print(f"    ⚠️  {module_id}: concept block not found — using full master data")

    feedback_block = format_validation_feedback(val_result)

    prompt = (
        splitter_prompt
        + "\n\n"
        + feedback_block
        + "\n\n"
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + "MASTER DATA FOR THIS CONCEPT:\n"
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + concept_text
    )

    response = None
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=get_vertex_safe_schema(ModuleList),
            ),
        )

        data = json.loads(response.text)
        if isinstance(data, list):
            data = {"modules": data}

        validated = ModuleList.model_validate(data)

        if not validated.modules:
            raise ValueError("Empty modules list returned by model")

        target = next(
            (m for m in validated.modules if m.module_id == module_id),
            validated.modules[0],
        )

        out_path = os.path.join(MODULES_FOLDER, chapter_name, f"{module_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(target.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)

        clear_validation_result(chapter_name, module_id)
        print(f"    ✓  {module_id}  saved")

    except Exception as e:
        raw = response.text if response else "no response"
        print(f"    ✗  {module_id}  FAILED: {e}")
        err_path = os.path.join(MODULES_FOLDER, chapter_name, f"{module_id}_REGEN_ERROR.txt")
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"ERROR: {e}\n\nRAW RESPONSE:\n{raw}")


# ─────────────────────────────────────────
# PROCESS ONE CHAPTER
# ─────────────────────────────────────────

def process_chapter(
    chapter_name: str,
    entries: list[dict],          # [{ module_id, val_result }, ...]
    splitter_prompt: str,
    model: GenerativeModel,
):
    master_path = os.path.join(MASTER_FOLDER, f"{chapter_name}.txt")
    if not os.path.exists(master_path):
        print(f"\n  ❌ [{chapter_name}] master data not found: {master_path}")
        return

    with open(master_path, "r", encoding="utf-8") as f:
        master_data_full = f.read()

    print(f"\n[{chapter_name}] Regenerating {len(entries)} module(s):")
    for entry in entries:
        regenerate_one_module(
            chapter_name,
            entry["module_id"],
            entry["val_result"],
            master_data_full,
            splitter_prompt,
            model,
        )


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

async def main():
    failed = collect_failed_modules()

    if not failed:
        print("\n✅ Nothing to regenerate — all validated modules are READY_TO_GO.")
        return

    total = sum(len(entries) for entries in failed.values())
    print(f"\n{'─'*55}")
    print(f"TO REGENERATE: {total} module(s) across {len(failed)} chapter(s)")
    for ch, entries in failed.items():
        print(f"  {ch}: {[e['module_id'] for e in entries]}")
    print(f"{'─'*55}")

    if DRY_RUN:
        print("\n⚠️  DRY_RUN = True — nothing was changed.")
        print("   Set DRY_RUN = False in CONFIG to actually regenerate.\n")
        return

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-flash")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        splitter_prompt = f.read()

    chapters = dict(failed)
    if DEBUG:
        first_ch = next(iter(chapters))
        chapters = {first_ch: chapters[first_ch][:1]}
        print(f"\n[DEBUG] Processing only: {chapters}")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, process_chapter, ch, entries, splitter_prompt, model)
            for ch, entries in chapters.items()
        ]
        await asyncio.gather(*tasks)

    print("\n✅ Done. Run validate_modules2.py to re-validate regenerated modules.")


if __name__ == "__main__":
    asyncio.run(main())