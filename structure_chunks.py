import json
import asyncio
from pathlib import Path
from typing import List, Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field

# ============================================================
# CONFIG
# ============================================================

CHUNKS_DIR     = Path("chunks")
OUTPUT_DIR     = Path("chunks_structured")
PROMPT_PATH    = Path("prompts/generate_chunks.txt")
MODEL_NAME     = "gpt-4.1"
TEMPERATURE    = 0.1
MAX_CONCURRENT = 5
DEBUG          = False

client = OpenAI(api_key="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")

# ============================================================
# SCHEMA
# ============================================================

class Transition(BaseModel):
    in_: str = Field(alias="in")
    out: str
    duration_ms: int
    model_config = {"populate_by_name": True}

class LayoutConfig(BaseModel):
    template: str
    text_zone: str
    visual_zone: str
    transition: Transition

class TTS(BaseModel):
    read_field: str
    language: str
    sync_mode: Literal["chunk", "per_step"]

class KeyPoint(BaseModel):
    label: str
    x: float
    y: float

class SpecialCell(BaseModel):
    row: int
    col: int
    note: str

class Visual(BaseModel):
    type: str
    render_target: Optional[str] = None
    description: Optional[str] = None
    id: Optional[str] = None
    graph_type: Optional[str] = None
    concept_latex: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    x_range: Optional[List[float]] = None
    y_range: Optional[List[float]] = None
    key_points: Optional[List[KeyPoint]] = None
    annotations: Optional[List[str]] = None
    exam_relevance: Optional[str] = None
    cannot_be_expressed_in_text: Optional[bool] = None
    purpose: Optional[str] = None
    headers: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None
    special_cells: Optional[List[SpecialCell]] = None

class ExamTip(BaseModel):
    question_pattern: str
    skill_tested: str
    distractor: str
    why_distractor_works: str

class PedagogyNote(BaseModel):
    misconception: str
    why_it_occurs: str
    correction: str
    exam_risk: str

class SolutionStep(BaseModel):
    step: int
    action_spoken: str
    action_display: str
    justification: str

class Part(BaseModel):
    part_label: str
    solution_steps: List[SolutionStep]
    part_answer: str
    part_answer_display: str

class DerivationStep(BaseModel):
    step: int
    action_spoken: str
    action_display: str
    justification: str

class ResultFormula(BaseModel):
    name: str
    latex: str
    spoken: str

class FormulaUsed(BaseModel):
    name: str
    latex: str
    spoken: str

class GivenItem(BaseModel):
    variable: str
    value: str
    meaning: str

class CoverageSummary(BaseModel):
    definitions_covered: int
    formulas_covered: int
    theorems_covered: int
    properties_covered: int
    worked_examples_covered: int
    length_problems_covered: int

class Chunk(BaseModel):
    chunk_id: str
    type: str
    layout: str
    layout_config: LayoutConfig
    tts: TTS
    slide_title: str
    script: Optional[str] = None
    script_display: Optional[str] = None
    display_bullets: Optional[List[str]] = None
    prerequisites_display: Optional[List[str]] = None
    next_modules: Optional[List[str]] = None
    coverage_summary: Optional[CoverageSummary] = None
    visual: Optional[Visual] = None
    exam_tip: Optional[ExamTip] = None
    pedagogy_note: Optional[PedagogyNote] = None
    what_is_derived: Optional[str] = None
    what_is_derived_display: Optional[str] = None
    derivation_steps: Optional[List[DerivationStep]] = None
    result_formula: Optional[ResultFormula] = None
    source_formula_name: Optional[str] = None
    difficulty: Optional[str] = None
    problem_statement: Optional[str] = None
    problem_statement_display: Optional[str] = None
    solution_steps: Optional[List[SolutionStep]] = None
    final_answer: Optional[str] = None
    final_answer_display: Optional[str] = None
    common_trap: Optional[str] = None
    source_example_id: Optional[str] = None
    parts: Optional[List[Part]] = None
    given: Optional[List[GivenItem]] = None
    unknown: Optional[str] = None
    formula_used: Optional[FormulaUsed] = None
    boundary_check: Optional[str] = None
    boundary_check_display: Optional[str] = None

class ChunkFile(BaseModel):
    module_id: str
    module_title: str
    chapter_title: str
    class_: str = Field(alias="class")
    chapter: str
    domain: str
    target_exam: str
    total_chunks: int
    coverage_warning: Optional[str] = None
    chunks: List[Chunk] = Field(min_length=1, max_length=40)
    model_config = {"populate_by_name": True}

# ============================================================
# LOAD PROMPT
# ============================================================

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# ============================================================
# OPENAI CALL — matches reference: client.responses.parse + text_format
# ============================================================

def structure_chunk_blocking(raw_json: str, module_id: str) -> ChunkFile:

    prompt = f"""{prompt_template}

The following is a raw chunk JSON generated by Gemini for module: {module_id}
Structure it strictly into the schema. Do NOT change any content, script, spoken text, or math.
Only organise into the correct fields.

RAW CHUNK JSON:
{raw_json}
"""

    response = client.responses.parse(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        input=[
            {
                "role": "system",
                "content": "You are a formatting engine. Do not modify any content. Return strictly valid JSON matching the provided schema."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        text_format=ChunkFile,
    )

    return response.output_parsed

# ============================================================
# PROCESS ONE CHUNK FILE
# ============================================================

async def process_chunk_file(chunk_path: Path, semaphore: asyncio.Semaphore):

    async with semaphore:

        relative    = chunk_path.relative_to(CHUNKS_DIR)
        output_path = OUTPUT_DIR / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"⏭  Skipped: {chunk_path}")
            return

        print(f"🚀 Structuring: {chunk_path}")

        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                raw_json = f.read()

            raw       = json.loads(raw_json)
            module_id = raw.get("module_id", chunk_path.stem)

            structured = await asyncio.to_thread(
                structure_chunk_blocking,
                raw_json,
                module_id
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    structured.model_dump(by_alias=True, exclude_none=True),
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(f"✅ Saved: {output_path}")

        except Exception as e:
            error_path = output_path.with_suffix(".ERROR.txt")
            error_path.write_text(str(e), encoding="utf-8")
            print(f"❌ Error: {chunk_path} → {e}")

# ============================================================
# MAIN
# ============================================================

async def main():

    chunk_files = sorted(CHUNKS_DIR.rglob("*.json"))

    if not chunk_files:
        print("No chunk files found in chunks/")
        return

    if DEBUG:
        chunk_files = chunk_files[:2]

    print(f"📦 Found {len(chunk_files)} chunk files\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks     = [process_chunk_file(p, semaphore) for p in chunk_files]
    await asyncio.gather(*tasks)

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())