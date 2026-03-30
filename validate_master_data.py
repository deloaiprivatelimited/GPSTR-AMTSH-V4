import os
import asyncio
import json
import vertexai
from typing import List
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# -----------------------------
# CONFIG
# -----------------------------

PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"

PDF_FOLDER = "merged"
MASTER_FOLDER = "master_data"
OUTPUT_FOLDER = "validation_reports"

PROMPT_PATH = "prompts/master_data_validation.txt"

MAX_WORKERS = 2
DEBUG = False

# -----------------------------
# INIT
# -----------------------------

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# SCHEMA DEFINITION
# -----------------------------

class LMSValidationReport(BaseModel):
    coverage_status: str
    subject_accuracy: str
    concept_integrity: str
    diagram_integrity: str
    worked_example_integrity: str
    formula_integrity: str
    structural_compliance: str
    critical_issues: List[str]
    major_issues: List[str]
    minor_issues: List[str]
    release_recommendation: str

# -----------------------------
# LOAD PROMPT
# -----------------------------

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    validation_prompt = f.read()

# -----------------------------
# PROCESS SINGLE PDF
# -----------------------------

def validate_pdf(pdf_file):

    name = os.path.splitext(pdf_file)[0]

    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    master_path = os.path.join(MASTER_FOLDER, f"{name}.txt")

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{name}_validation.json"
    )

    if not os.path.exists(master_path):
        print("⚠ Missing master_data:", name)
        return

    if os.path.exists(output_path):
        # print("Skipping:", name)
        return

    print("Validating:", name)

    with open(master_path, "r", encoding="utf-8") as f:
        master_text = f.read()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_part = Part.from_data(
        data=pdf_bytes,
        mime_type="application/pdf"
    )

    prompt = f"""
{validation_prompt}

----- SOURCE TEXTBOOK PDF -----
"""

    # Enforce the Pydantic schema here
    response = model.generate_content(
        [prompt, pdf_part, master_text],
        generation_config=GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=LMSValidationReport.model_json_schema()
        )
    )

    try:
        # Validate the response string directly back into the Pydantic object
        report = LMSValidationReport.model_validate_json(response.text)
        
        # Dump it to a dictionary for safe saving
        parsed_data = report.model_dump()
        
    except Exception as e:
        print(f"⚠ Failed to parse/validate JSON for {name}: {e}")
        # Fallback to save raw response if it breaks the schema
        parsed_data = {"raw_response": response.text, "error": str(e)}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)

    print("Saved:", name)


# -----------------------------
# ASYNC RUNNER
# -----------------------------

async def main():

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER)
        if f.endswith(".pdf")
    ]

    pdf_files.sort()

    if DEBUG:
        pdf_files = pdf_files[:1]

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        tasks = []

        for pdf_file in pdf_files:

            task = loop.run_in_executor(
                executor,
                validate_pdf,
                pdf_file
            )

            tasks.append(task)

        await asyncio.gather(*tasks)

    print("\n🚀 All validations completed")


# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    asyncio.run(main())