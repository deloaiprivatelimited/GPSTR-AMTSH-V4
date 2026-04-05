import os
import json
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# -----------------------------
# CONFIG
# -----------------------------
LOCATION = "us-central1"

CREDENTIALS_DIR = Path("claude_works/credentials")
PDF_FOLDER = Path("science_textbooks/split")
PROMPT_PATH = Path("prompts_v2/1_extract_text_science.txt")
OUTPUT_FOLDER = Path("claude_works/extracted_text_science")

THREADS_PER_PROJECT = 4
DEBUG = False   # True → only 2 PDFs total

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD PROMPT
# -----------------------------
extract_prompt = PROMPT_PATH.read_text(encoding="utf-8")

# -----------------------------
# MULTI-PROJECT INIT (same pattern as generate_podcast.py)
# -----------------------------
USE_ONLY = ["disco-vista-355914-acd5b82a7f2d.json"]

PROJECTS = []
for cred_file in sorted(os.listdir(CREDENTIALS_DIR)):
    if not cred_file.endswith(".json"):
        continue
    if USE_ONLY and cred_file not in USE_ONLY:
        continue
    cred_path = CREDENTIALS_DIR / cred_file
    creds = service_account.Credentials.from_service_account_file(str(cred_path))
    project_id = json.load(open(cred_path))["project_id"]
    PROJECTS.append({"project_id": project_id, "credentials": creds})
    print(f"  Loaded project: {project_id}")

print(f"Total projects: {len(PROJECTS)}")

# Thread-local model storage (avoids vertexai.init global state conflicts)
_thread_local = threading.local()

def get_model(project_idx):
    """Get or create a thread-local model for the given project."""
    key = f"model_{project_idx}"
    if not hasattr(_thread_local, key):
        p = PROJECTS[project_idx]
        vertexai.init(project=p["project_id"], location=LOCATION, credentials=p["credentials"])
        setattr(_thread_local, key, GenerativeModel("gemini-2.5-pro"))
    return getattr(_thread_local, key)

# -----------------------------
# COLLECT ALL CHAPTER PDFs
# -----------------------------
def collect_pdfs():
    """Collect all chapter PDFs from split/std_X/ folders."""
    pdf_list = []
    for std_dir in sorted(os.listdir(PDF_FOLDER)):
        std_path = PDF_FOLDER / std_dir
        if not std_path.is_dir() or not std_dir.startswith("std_"):
            continue
        for chapter_file in sorted(os.listdir(std_path)):
            if chapter_file.endswith(".pdf"):
                pdf_list.append((std_dir, chapter_file))
    return pdf_list

# -----------------------------
# SPLIT LIST INTO N BATCHES
# -----------------------------
def split_into_batches(items, n):
    """Split items into n roughly equal batches."""
    batch_size = math.ceil(len(items) / n)
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# -----------------------------
# PROCESS SINGLE PDF (with retry)
# -----------------------------
def process_pdf(project_idx, std_dir, chapter_file):
    p = PROJECTS[project_idx]
    project_id = p["project_id"]

    pdf_path = PDF_FOLDER / std_dir / chapter_file
    std_num = std_dir.replace("std_", "")
    chapter_name = os.path.splitext(chapter_file)[0]

    output_name = f"{std_dir}_{chapter_name}"
    output_path = OUTPUT_FOLDER / f"{output_name}.txt"

    if output_path.exists():
        print(f"  [{project_id}] Skipping: {output_name}")
        return

    print(f"  [{project_id}] Processing: {output_name}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_part = Part.from_data(
        data=pdf_bytes,
        mime_type="application/pdf"
    )

    prompt = f"""SOURCE_FILE: {std_dir}/{chapter_file} (Class {std_num})

{extract_prompt}"""

    max_retries = 10
    for attempt in range(max_retries):
        try:
            model = get_model(project_idx)
            response = model.generate_content(
                [pdf_part, prompt],
                generation_config={
                    "temperature": 0.1
                }
            )

            output_path.write_text(response.text, encoding="utf-8")
            print(f"  [{project_id}] Saved: {output_name}")
            return

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"  [{project_id}] 429 rate limit on {output_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif any(code in err for code in ["500", "502", "503", "504", "INTERNAL", "UNAVAILABLE"]):
                wait = 30 * (attempt + 1)
                print(f"  [{project_id}] Server error on {output_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "DEADLINE_EXCEEDED" in err or "timeout" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"  [{project_id}] Timeout on {output_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                if attempt < 5:
                    wait = 15 * (attempt + 1)
                    print(f"  [{project_id}] Error on {output_name}: {err[:100]}, retrying in {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    error_path = OUTPUT_FOLDER / f"{output_name}_ERROR.txt"
                    error_path.write_text(err, encoding="utf-8")
                    print(f"  [{project_id}] FAILED: {output_name} -- {err[:100]}")
                    return

    # All retries exhausted
    error_path = OUTPUT_FOLDER / f"{output_name}_ERROR.txt"
    error_path.write_text(f"Max retries ({max_retries}) exhausted: {err}", encoding="utf-8")
    print(f"  [{project_id}] FAILED (max retries): {output_name}")

# -----------------------------
# WORKER WRAPPER
# -----------------------------
def process_pdf_wrapper(args):
    project_idx, std_dir, chapter_file = args
    return process_pdf(project_idx, std_dir, chapter_file)

# -----------------------------
# RUN ONE PROJECT BATCH
# -----------------------------
def run_project_batch(project_idx, batch):
    p = PROJECTS[project_idx]
    tasks = [(project_idx, std_dir, chapter_file) for std_dir, chapter_file in batch]

    print(f"\n[{p['project_id']}] Starting batch of {len(batch)} PDFs with {THREADS_PER_PROJECT} threads")

    with ThreadPoolExecutor(max_workers=THREADS_PER_PROJECT) as executor:
        executor.map(process_pdf_wrapper, tasks)

    print(f"[{p['project_id']}] Batch complete")

# -----------------------------
# MAIN
# -----------------------------
def main():
    all_pdfs = collect_pdfs()

    if DEBUG:
        all_pdfs = all_pdfs[:2]

    num_projects = len(PROJECTS)
    batches = split_into_batches(all_pdfs, num_projects)

    print(f"\n{'='*60}")
    print(f"SCIENCE TEXT EXTRACTION — {num_projects} projects")
    print(f"{'='*60}")
    print(f"Total PDFs: {len(all_pdfs)}")
    print(f"Threads per project: {THREADS_PER_PROJECT}")
    print(f"Max parallel extractions: {num_projects * THREADS_PER_PROJECT}\n")

    for i, batch in enumerate(batches):
        print(f"  Project {i} ({PROJECTS[i]['project_id']}): {len(batch)} PDFs")

    # Run each project batch in parallel
    with ThreadPoolExecutor(max_workers=num_projects) as executor:
        futures = []
        for i, batch in enumerate(batches):
            if batch:
                futures.append(executor.submit(run_project_batch, i, batch))
        for f in futures:
            f.result()

    print("\nAll science PDFs processed!")

if __name__ == "__main__":
    main()
