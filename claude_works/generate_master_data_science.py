"""
Generate structured master data for science chapters.
- Watches extracted_text_science/ for new .txt files (polling)
- Sends both extracted text + original PDF to Gemini for cross-reference
- Saves master data to claude_works/master_data_science/
- Uses podcast-tts credential, 4 parallel threads
- Auto-polls every 30s for new files from the extraction script
"""
import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
LOCATION = "us-central1"

CREDENTIALS_DIR = Path("claude_works/credentials")
USE_ONLY = ["podcast-tts-488313-4868c70c14ad.json"]

TEXT_FOLDER = Path("claude_works/extracted_text_science")
PDF_FOLDER = Path("science_textbooks/split")
PROMPT_PATH = Path("prompts_v2/2_master_data_science.txt")
OUTPUT_FOLDER = Path("claude_works/master_data_science")

THREADS = 4
POLL_INTERVAL = 30   # seconds between polls for new files
MAX_POLLS = 200      # stop after this many empty polls (~100 min)
DEBUG = False        # True → only 1 file

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────
# LOAD PROMPT
# ─────────────────────────────────────
master_prompt = PROMPT_PATH.read_text(encoding="utf-8")

# ─────────────────────────────────────
# PROJECT INIT
# ─────────────────────────────────────
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

# Thread-local model
_thread_local = threading.local()

def get_model(project_idx=0):
    key = f"model_{project_idx}"
    if not hasattr(_thread_local, key):
        p = PROJECTS[project_idx]
        vertexai.init(project=p["project_id"], location=LOCATION, credentials=p["credentials"])
        setattr(_thread_local, key, GenerativeModel("gemini-2.5-pro"))
    return getattr(_thread_local, key)

# ─────────────────────────────────────
# MAP txt name → PDF path
# ─────────────────────────────────────
def txt_to_pdf_path(txt_name):
    """Convert 'std_6_chapter_1' → 'science_textbooks/split/std_6/chapter_1.pdf'"""
    # txt_name like: std_6_chapter_1
    parts = txt_name.split("_", 2)  # ['std', '6', 'chapter_1']
    if len(parts) < 3:
        return None
    std_dir = f"{parts[0]}_{parts[1]}"  # std_6
    chapter_file = f"{parts[2]}.pdf"     # chapter_1.pdf
    pdf_path = PDF_FOLDER / std_dir / chapter_file
    return pdf_path if pdf_path.exists() else None

def parse_chapter_info(txt_name):
    """Extract class number and chapter from txt_name like 'std_6_chapter_1'"""
    parts = txt_name.split("_", 2)
    if len(parts) < 3:
        return None, None
    std_num = parts[1]
    chapter = parts[2]  # chapter_1
    return std_num, chapter

# ─────────────────────────────────────
# FIND PENDING FILES
# ─────────────────────────────────────
def find_pending():
    """Find .txt files in extracted_text_science that don't have master data yet."""
    pending = []
    if not TEXT_FOLDER.exists():
        return pending
    for f in sorted(os.listdir(TEXT_FOLDER)):
        if not f.endswith(".txt") or f.endswith("_ERROR.txt"):
            continue
        name = f.replace(".txt", "")
        output_path = OUTPUT_FOLDER / f"{name}.txt"
        error_path = OUTPUT_FOLDER / f"{name}_ERROR.txt"
        if not output_path.exists() and not error_path.exists():
            pending.append(name)
    return pending

# ─────────────────────────────────────
# PROCESS ONE CHAPTER
# ─────────────────────────────────────
def process_chapter(txt_name):
    project_id = PROJECTS[0]["project_id"]
    txt_path = TEXT_FOLDER / f"{txt_name}.txt"
    output_path = OUTPUT_FOLDER / f"{txt_name}.txt"

    # Skip if already done
    if output_path.exists():
        return "skipped"

    std_num, chapter = parse_chapter_info(txt_name)
    pdf_path = txt_to_pdf_path(txt_name)

    print(f"  [{project_id}] Processing: {txt_name} (Class {std_num})")

    # Read extracted text
    extracted_text = txt_path.read_text(encoding="utf-8")

    # Build prompt parts
    prompt_parts = []

    # 1. PDF (if available) for cross-reference
    if pdf_path:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_part = Part.from_data(data=pdf_bytes, mime_type="application/pdf")
        prompt_parts.append(pdf_part)

    # 2. Text prompt with extracted content
    chapter_code = txt_name.upper().replace("-", "_")
    full_prompt = f"""CHAPTER_CODE: {txt_name}
CLASS: {std_num}
CHAPTER: {chapter}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTED TEXT (PRIMARY SOURCE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{extracted_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{"The PDF of this chapter is also provided. Use it to cross-check the extracted text and add anything that was missed." if pdf_path else "No PDF available for this chapter. Use only the extracted text."}

{master_prompt}"""

    prompt_parts.append(full_prompt)

    # Call Gemini with retry
    max_retries = 10
    last_err = ""
    for attempt in range(max_retries):
        try:
            model = get_model(0)
            response = model.generate_content(
                prompt_parts,
                generation_config={"temperature": 0.1}
            )

            output_path.write_text(response.text, encoding="utf-8")
            print(f"  [{project_id}] Saved: {txt_name}")
            return "ok"

        except Exception as e:
            last_err = str(e)
            if "429" in last_err or "RESOURCE_EXHAUSTED" in last_err:
                wait = 60 * (attempt + 1)
                print(f"  [{project_id}] 429 on {txt_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif any(code in last_err for code in ["500", "502", "503", "504", "INTERNAL", "UNAVAILABLE"]):
                wait = 30 * (attempt + 1)
                print(f"  [{project_id}] Server error on {txt_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "DEADLINE_EXCEEDED" in last_err or "timeout" in last_err.lower():
                wait = 30 * (attempt + 1)
                print(f"  [{project_id}] Timeout on {txt_name}, waiting {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                if attempt < 5:
                    wait = 15 * (attempt + 1)
                    print(f"  [{project_id}] Error on {txt_name}: {last_err[:100]}, retrying in {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    break

    # All retries failed
    error_path = OUTPUT_FOLDER / f"{txt_name}_ERROR.txt"
    error_path.write_text(f"Max retries ({max_retries}) exhausted: {last_err}", encoding="utf-8")
    print(f"  [{project_id}] FAILED: {txt_name}")
    return "error"

# ─────────────────────────────────────
# PROCESS BATCH
# ─────────────────────────────────────
def process_batch(pending):
    """Process a list of pending txt_names with THREADS parallel workers."""
    results = {"ok": 0, "error": 0, "skipped": 0}

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(process_chapter, name): name for name in pending}
        for future in as_completed(futures):
            name = futures[future]
            try:
                status = future.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                print(f"  Unexpected error on {name}: {e}")
                results["error"] += 1

    return results

# ─────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────
def print_summary():
    total = 0
    errors = 0
    for f in os.listdir(OUTPUT_FOLDER):
        if f.endswith("_ERROR.txt"):
            errors += 1
        elif f.endswith(".txt"):
            total += 1

    available_txt = len([f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt") and not f.endswith("_ERROR.txt")])

    print(f"\n{'='*60}")
    print(f"MASTER DATA SCIENCE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Extracted texts available: {available_txt}")
    print(f"  Master data generated: {total}")
    print(f"  Errors: {errors}")
    print(f"  Remaining: {available_txt - total - errors}")

# ─────────────────────────────────────
# MAIN — POLLING LOOP
# ─────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"SCIENCE MASTER DATA GENERATION")
    print(f"{'='*60}")
    print(f"Credential: {PROJECTS[0]['project_id']}")
    print(f"Threads: {THREADS}")
    print(f"Polling interval: {POLL_INTERVAL}s")
    print(f"Watching: {TEXT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}\n")

    total_processed = 0
    empty_polls = 0

    while empty_polls < MAX_POLLS:
        pending = find_pending()

        if DEBUG:
            pending = pending[:1]

        if not pending:
            empty_polls += 1
            if empty_polls == 1:
                print(f"\nNo pending files. Polling every {POLL_INTERVAL}s for new extracted texts...")
            elif empty_polls % 10 == 0:
                print(f"  Still waiting... ({empty_polls} polls, {empty_polls * POLL_INTERVAL}s elapsed)")
            time.sleep(POLL_INTERVAL)
            continue

        # Reset empty poll counter when we find work
        empty_polls = 0
        print(f"\nFound {len(pending)} pending chapters: {pending[:5]}{'...' if len(pending) > 5 else ''}")

        results = process_batch(pending)
        total_processed += results.get("ok", 0)

        print(f"\nBatch done — ok:{results.get('ok',0)} error:{results.get('error',0)} skipped:{results.get('skipped',0)}")
        print(f"Total processed so far: {total_processed}")

        if DEBUG:
            break

    print_summary()
    print("\nDone!")

if __name__ == "__main__":
    main()
