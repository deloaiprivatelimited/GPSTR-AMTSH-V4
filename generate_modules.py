import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID     = "project-6565cf16-a3d4-4f6e-935"
LOCATION       = "us-central1"
MASTER_FOLDER  = "master_data"
PROMPT_PATH    = "prompts/module_splitter.txt"
OUTPUT_FOLDER  = "modules"
MAX_WORKERS    = 4
DEBUG          = False

# -----------------------------
# PYDANTIC MODELS
# -----------------------------

class KeyPoint(BaseModel):
    label: str
    x: float
    y: float

class DiagramVisualAid(BaseModel):
    """
    Mirrors the Diagram JSON format defined in master_data.txt.
    Used when a concept requires a plotly/svg/canvas rendering.
    """
    type: str = "diagram"
    id: str
    render_target: str                          # "plotly" | "svg_template" | "canvas"
    graph_type: str                             # "parabola" | "line" | "geometric_figure" | "number_line" | "venn" | "flowchart"
    concept_latex: str
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    x_range: Optional[List[float]] = None
    y_range: Optional[List[float]] = None
    key_points: List[KeyPoint] = Field(default_factory=list)
    annotations: List[str] = Field(default_factory=list)
    exam_relevance: str                         # "MCQ" | "2-mark" | "5-mark"
    cannot_be_expressed_in_text: bool = True

class SpecialCell(BaseModel):
    row: int
    col: int
    note: str

class TableVisualAid(BaseModel):
    """
    Mirrors the Table JSON format defined in master_data.txt.
    Used when a concept requires an html_table or xlsx rendering.
    """
    type: str = "table"
    id: str
    render_target: str                          # "html_table" | "xlsx"
    purpose: str
    headers: List[str]
    rows: List[List[str]]
    special_cells: List[SpecialCell] = Field(default_factory=list)
    exam_relevance: str                         # "MCQ" | "2-mark" | "5-mark"
    cannot_be_expressed_in_text: bool = True

class Definition(BaseModel):
    term: str
    definition: str

class Formula(BaseModel):
    formula_name: str
    latex: str                                  # character-perfect copy from master data
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
    difficulty: str                             # "basic" | "intermediate" | "advanced"
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
    # FIX: Full diagram/table JSON preserved — not flattened to {type, description, label}
    # Vertex AI does not support Union types in response_schema, so we use a combined
    # model that covers all fields from both DiagramVisualAid and TableVisualAid.
    # Fields not applicable to a given entry are left as None/empty.
    visual_aids: List["VisualAidUnified"] = Field(default_factory=list)

class VisualAidUnified(BaseModel):
    """
    Single unified model covering both DiagramVisualAid and TableVisualAid fields.
    Vertex AI response_schema does not support Union/discriminated unions, so we
    merge both schemas here. The `type` field ("diagram" | "table") tells consumers
    which fields are relevant at render time.
    """
    type: str                                   # "diagram" | "table"
    id: str
    render_target: str                          # "plotly"|"svg_template"|"canvas"|"html_table"|"xlsx"
    exam_relevance: str                         # "MCQ" | "2-mark" | "5-mark"
    cannot_be_expressed_in_text: bool = True

    # --- Diagram-only fields ---
    graph_type: Optional[str] = None           # "parabola"|"line"|"geometric_figure"|"number_line"|"venn"|"flowchart"
    concept_latex: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    x_range: Optional[List[float]] = None
    y_range: Optional[List[float]] = None
    key_points: List[KeyPoint] = Field(default_factory=list)
    annotations: List[str] = Field(default_factory=list)

    # --- Table-only fields ---
    purpose: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    special_cells: List[SpecialCell] = Field(default_factory=list)

# Rebuild Theory so it sees the now-defined VisualAidUnified
Theory.model_rebuild()

class ExamIntelligence(BaseModel):
    gpstr_weightage: str                        # "high" | "medium" | "low"
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

# -----------------------------
# VERTEX DEEP CLEANER
# -----------------------------

def get_vertex_safe_schema(model_class):
    full_schema = model_class.model_json_schema()
    definitions = full_schema.get("$defs", {})

    def resolve_and_clean(node):
        if not isinstance(node, dict):
            return node

        # Resolve $ref
        if "$ref" in node:
            ref_key = node["$ref"].split("/")[-1]
            return resolve_and_clean(definitions[ref_key])

        # Handle anyOf (Optional fields) — keep first non-null type
        if "anyOf" in node:
            real_options = [opt for opt in node["anyOf"] if opt.get("type") != "null"]
            if real_options:
                return resolve_and_clean(real_options[0])

        # Recursive cleaning
        if "properties" in node:
            node["properties"] = {k: resolve_and_clean(v) for k, v in node["properties"].items()}
        if "items" in node:
            node["items"] = resolve_and_clean(node["items"])

        # Remove fields Vertex AI rejects
        node.pop("title", None)
        node.pop("default", None)
        node.pop("additionalProperties", None)
        node.pop("$defs", None)

        return node

    return resolve_and_clean(full_schema)

# -----------------------------
# RUNNER LOGIC
# -----------------------------

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-flash")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    splitter_prompt = f.read()

def save_modules(chapter_name: str, modules: List[Module]):
    chapter_folder = os.path.join(OUTPUT_FOLDER, chapter_name)
    os.makedirs(chapter_folder, exist_ok=True)
    for m in modules:
        path = os.path.join(chapter_folder, f"{m.module_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)

def process_master_file(txt_file: str):
    chapter_name = os.path.splitext(txt_file)[0]
    chapter_folder = os.path.join(OUTPUT_FOLDER, chapter_name)

    if os.path.exists(chapter_folder) and len(os.listdir(chapter_folder)) > 0:
        return

    print(f"Working on: {chapter_name}")

    with open(os.path.join(MASTER_FOLDER, txt_file), "r", encoding="utf-8") as f:
        master_data = f.read()

    prompt = f"{splitter_prompt}\n\nINPUT DATA:\n{master_data}"

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=get_vertex_safe_schema(ModuleList)
            )
        )

        data = json.loads(response.text)
        validated = ModuleList.model_validate(data)
        save_modules(chapter_name, validated.modules)
        print(f"Success: {chapter_name}")

    except Exception as e:
        error_file = os.path.join(OUTPUT_FOLDER, f"{chapter_name}_ERROR.txt")
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"ERROR: {str(e)}\n\nRAW RESPONSE:\n{response.text if 'response' in locals() else 'None'}")
        print(f"Fail: {chapter_name}. Check {error_file}")

async def main():
    txt_files = sorted([f for f in os.listdir(MASTER_FOLDER) if f.endswith(".txt")])
    if DEBUG: txt_files = txt_files[:1]

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [loop.run_in_executor(executor, process_master_file, f) for f in txt_files]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())