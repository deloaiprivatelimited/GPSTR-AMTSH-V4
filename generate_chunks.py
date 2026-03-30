import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID     = "project-6565cf16-a3d4-4f6e-935"
LOCATION       = "us-central1"
MODULES_FOLDER = "modules"
PROMPT_PATH    = "prompts/generate_chunks.txt"
OUTPUT_FOLDER  = "chunks"
MAX_WORKERS    = 4
DEBUG          = True

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    chunk_prompt = f.read()

# -----------------------------
# PROCESS ONE MODULE
# -----------------------------
def process_module(module_path: str, output_path: str):

    if os.path.exists(output_path):
        return

    with open(module_path, "r", encoding="utf-8") as f:
        module_json = f.read()

    prompt = chunk_prompt.replace("[INSERT MODULE JSON HERE]", module_json)

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"  ✓ {os.path.basename(output_path)}")

    except Exception as e:
        error_path = output_path.replace(".json", "_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"  ✗ {os.path.basename(output_path)} — {e}")

# -----------------------------
# COLLECT ALL MODULE JOBS
# -----------------------------
def collect_jobs():
    jobs = []

    chapter_dirs = sorted([
        d for d in os.listdir(MODULES_FOLDER)
        if os.path.isdir(os.path.join(MODULES_FOLDER, d))
    ])

    for chapter in chapter_dirs:
        chapter_input  = os.path.join(MODULES_FOLDER, chapter)
        chapter_output = os.path.join(OUTPUT_FOLDER, chapter)
        os.makedirs(chapter_output, exist_ok=True)

        module_files = sorted([
            f for f in os.listdir(chapter_input)
            if f.endswith(".json")
        ])

        for mf in module_files:
            module_path = os.path.join(chapter_input, mf)
            output_path = os.path.join(chapter_output, mf)
            jobs.append((module_path, output_path))

    return jobs

# -----------------------------
# ASYNC RUNNER
# -----------------------------
async def main():
    jobs = collect_jobs()

    if not jobs:
        print("No modules found.")
        return

    if DEBUG:
        jobs = jobs[:2]

    print(f"Processing {len(jobs)} modules across {len(set(os.path.dirname(o) for _, o in jobs))} chapters\n")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, process_module, module_path, output_path)
            for module_path, output_path in jobs
        ]
        await asyncio.gather(*tasks)

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())