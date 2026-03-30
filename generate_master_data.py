import os
import asyncio
import vertexai
from concurrent.futures import ThreadPoolExecutor
from vertexai.generative_models import GenerativeModel, Part

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"

PDF_FOLDER = "merged"
PROMPT_PATH = "prompts/master_data.txt"
OUTPUT_FOLDER = "master_data"

MAX_WORKERS = 4
DEBUG = False   # True → only one PDF

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LOAD PROMPT
# -----------------------------
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    master_prompt = f.read()

# -----------------------------
# PROCESS SINGLE PDF
# -----------------------------
def process_pdf(pdf_file):

    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    name = os.path.splitext(pdf_file)[0]

    output_path = os.path.join(OUTPUT_FOLDER, f"{name}.txt")

    if os.path.exists(output_path):
        print("Skipping:", name)
        return

    print("Processing:", name)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_part = Part.from_data(
        data=pdf_bytes,
        mime_type="application/pdf"
    )

    prompt = f"""
SOURCE_FILE: {pdf_file}

{master_prompt}
"""

    response = model.generate_content(
        [pdf_part, prompt],
        generation_config={
            "temperature": 0.1
        }
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

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

    # DEBUG MODE
    if DEBUG:
        pdf_files = pdf_files[:1]

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        tasks = []

        for pdf_file in pdf_files:

            task = loop.run_in_executor(
                executor,
                process_pdf,
                pdf_file
            )

            tasks.append(task)

        await asyncio.gather(*tasks)

    print("All PDFs processed")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())