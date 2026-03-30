"""
Batch Generate Chunks using Vertex AI Batch Prediction
- Prepares all pending prompts as JSONL
- Uploads to GCS
- Submits batch job
- Polls for completion
- Downloads and saves results as separate chunk files
"""
import os
import json
import time
import vertexai
from google.cloud import storage
from vertexai.batch_prediction import BatchPredictionJob

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"
MODEL = "gemini-2.5-pro"
BUCKET_NAME = "gpstr-maths-batch-2026"

MODULES_FOLDER = "claude_works/modules"
CHUNKS_FOLDER = "claude_works/chunks"
BATCH_FOLDER = "claude_works/batch"

os.makedirs(BATCH_FOLDER, exist_ok=True)
os.makedirs(CHUNKS_FOLDER, exist_ok=True)

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
storage_client = storage.Client(project=PROJECT_ID)

# ═════════════════════════════════════════
# LOAD SHARED RULES + PROMPTS (from generate_chunks.py)
# ═════════════════════════════════════════

# Import prompts from generate_chunks.py
import importlib.util
spec = importlib.util.spec_from_file_location("gen_chunks", "claude_works/generate_chunks.py")
gen_mod = importlib.util.module_from_spec(spec)

# We need to prevent it from running vertexai.init and model creation
import unittest.mock
with unittest.mock.patch('vertexai.init'), \
     unittest.mock.patch('vertexai.generative_models.GenerativeModel'):
    spec.loader.exec_module(gen_mod)

SHARED_RULES = gen_mod.SHARED_RULES
INTRO_PROMPT = gen_mod.INTRO_PROMPT
DEFINITION_PROMPT = gen_mod.DEFINITION_PROMPT
FORMULA_PROMPT = gen_mod.FORMULA_PROMPT
THEOREM_PROMPT = gen_mod.THEOREM_PROMPT
PROPERTY_PROMPT = gen_mod.PROPERTY_PROMPT
WORKED_EXAMPLE_PROMPT = gen_mod.WORKED_EXAMPLE_PROMPT
RECAP_PROMPT = gen_mod.RECAP_PROMPT

def format_prompt(template):
    return template.replace("{shared_rules}", SHARED_RULES)

# ═════════════════════════════════════════
# STEP 1: BUILD ALL PENDING PROMPTS
# ═════════════════════════════════════════

def get_pending_calls():
    """Find all API calls that haven't been made yet"""
    pending = []

    for mc in sorted(os.listdir(MODULES_FOLDER)):
        mp = os.path.join(MODULES_FOLDER, mc)
        if not os.path.isdir(mp):
            continue

        for mf in sorted(os.listdir(mp)):
            if not mf.endswith(".json"):
                continue

            module_id = os.path.splitext(mf)[0]
            module_path = os.path.join(mp, mf)
            chunk_dir = os.path.join(CHUNKS_FOLDER, mc, module_id)

            # Load module
            with open(module_path, "r", encoding="utf-8") as f:
                module_data = json.load(f)

            theory = module_data.get("theory", {})
            definitions = theory.get("definitions", [])
            formulas = theory.get("formulas", [])
            theorems = theory.get("theorems", [])
            properties = theory.get("properties", [])
            visual_aids = theory.get("visual_aids", [])
            examples = module_data.get("worked_examples", [])
            exam_intel = module_data.get("exam_intelligence", {})

            prop_batches = [properties[i:i+3] for i in range(0, len(properties), 3)] if properties else []

            module_context = f"""
MODULE CONTEXT:
module_id: {module_id}
module_title: {module_data.get('module_title', '')}
chapter_title: {module_data.get('chapter_title', '')}
class: {module_data.get('class', '')}
domain: {module_data.get('domain', '')}
concept_summary: {theory.get('concept_summary', '')}
exam_intelligence: {json.dumps(exam_intel, ensure_ascii=False)}
visual_aids_in_module: {json.dumps(visual_aids, indent=2, ensure_ascii=False)}
"""

            # Track expected sequence number
            seq = 1

            def check_and_add(prompt, label, chunk_type, index):
                nonlocal seq
                expected_file = f"{seq:03d}_{chunk_type}_{index}.json"
                chunk_path = os.path.join(chunk_dir, expected_file) if os.path.isdir(chunk_dir) else ""

                if chunk_path and os.path.exists(chunk_path):
                    seq += 1  # skip existing
                    return

                pending.append({
                    "merge_code": mc,
                    "module_id": module_id,
                    "label": label,
                    "chunk_type": chunk_type,
                    "index": index,
                    "seq": seq,
                    "prompt": prompt
                })
                seq += 1

            # Intro
            prompt = format_prompt(INTRO_PROMPT) + module_context
            prompt += f"\nprerequisites: {json.dumps(module_data.get('prerequisites', []), ensure_ascii=False)}"
            check_and_add(prompt, f"{module_id}_intro", "intro", 0)

            # Definitions
            for i, defn in enumerate(definitions):
                prompt = format_prompt(DEFINITION_PROMPT) + module_context
                prompt += f"\nDefinition {i+1} of {len(definitions)}"
                prompt += f"\n\nDEFINITION TO TEACH:\n{json.dumps(defn, indent=2, ensure_ascii=False)}"
                check_and_add(prompt, f"{module_id}_def_{i}", "definition", i)

            # Theorems
            for i, thm in enumerate(theorems):
                prompt = format_prompt(THEOREM_PROMPT) + module_context
                prompt += f"\nTheorem {i+1} of {len(theorems)}"
                prompt += f"\n\nTHEOREM TO TEACH:\n{json.dumps(thm, indent=2, ensure_ascii=False)}"
                check_and_add(prompt, f"{module_id}_thm_{i}", "theorem", i)

            # Properties
            for i, batch in enumerate(prop_batches):
                prompt = format_prompt(PROPERTY_PROMPT) + module_context
                prompt += f"\nProperty batch {i+1} of {len(prop_batches)}"
                prompt += f"\n\nPROPERTIES TO TEACH:\n{json.dumps(batch, indent=2, ensure_ascii=False)}"
                check_and_add(prompt, f"{module_id}_prop_{i}", "property", i)

            # Formulas
            for i, formula in enumerate(formulas):
                prompt = format_prompt(FORMULA_PROMPT) + module_context
                prompt += f"\nFormula {i+1} of {len(formulas)}"
                prompt += f"\n\nFORMULA TO TEACH:\n{json.dumps(formula, indent=2, ensure_ascii=False)}"
                check_and_add(prompt, f"{module_id}_formula_{i}", "formula", i)

            # Examples
            for i, example in enumerate(examples):
                prompt = format_prompt(WORKED_EXAMPLE_PROMPT) + module_context
                prompt += f"\nExample {i+1} of {len(examples)}"
                prompt += f"\n\nWORKED EXAMPLE TO SOLVE:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
                check_and_add(prompt, f"{module_id}_ex_{i}", "example", i)

            # Recap
            prompt = format_prompt(RECAP_PROMPT) + module_context
            prompt += f"\nCovered: {len(definitions)} defs, {len(formulas)} formulas, "
            prompt += f"{len(theorems)} theorems, {len(properties)} properties, {len(examples)} examples"
            prompt += f"\nconnects_to: {json.dumps(exam_intel.get('connects_to', []), ensure_ascii=False)}"
            check_and_add(prompt, f"{module_id}_recap", "recap", 0)

    return pending

# ═════════════════════════════════════════
# STEP 2: CREATE JSONL + UPLOAD TO GCS
# ═════════════════════════════════════════

def create_and_upload_jsonl(pending):
    """Create JSONL file and upload to GCS"""

    jsonl_path = os.path.join(BATCH_FOLDER, "batch_input.jsonl")
    mapping_path = os.path.join(BATCH_FOLDER, "batch_mapping.json")

    # Create JSONL - each line is one request
    mapping = []
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(pending):
            request = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": item["prompt"]}]}],
                    "generation_config": {"temperature": 0.1}
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

            mapping.append({
                "index": i,
                "merge_code": item["merge_code"],
                "module_id": item["module_id"],
                "label": item["label"],
                "chunk_type": item["chunk_type"],
                "chunk_index": item["index"],
                "seq": item["seq"]
            })

    # Save mapping locally
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # Upload to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob("batch_input.jsonl")
    blob.upload_from_filename(jsonl_path)

    gcs_uri = f"gs://{BUCKET_NAME}/batch_input.jsonl"
    print(f"Uploaded {len(pending)} prompts to {gcs_uri}")
    print(f"Mapping saved to {mapping_path}")

    return gcs_uri, mapping

# ═════════════════════════════════════════
# STEP 3: SUBMIT BATCH JOB
# ═════════════════════════════════════════

def submit_batch_job(input_uri):
    """Submit batch prediction job"""

    output_uri = f"gs://{BUCKET_NAME}/batch_output/"

    job = BatchPredictionJob.submit(
        source_model=f"publishers/google/models/{MODEL}",
        input_dataset=input_uri,
        output_uri_prefix=output_uri,
    )

    print(f"Batch job submitted: {job.resource_name}")
    print(f"Output will be at: {output_uri}")

    # Save job info
    job_info = {
        "job_name": job.resource_name,
        "input_uri": input_uri,
        "output_uri": output_uri,
        "status": "SUBMITTED",
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(BATCH_FOLDER, "batch_job.json"), "w") as f:
        json.dump(job_info, f, indent=2)

    return job

# ═════════════════════════════════════════
# STEP 4: POLL FOR COMPLETION
# ═════════════════════════════════════════

def wait_for_job(job):
    """Poll until job completes"""
    print("\nWaiting for batch job to complete...")

    while True:
        job.refresh()
        state = job.state.name if hasattr(job.state, 'name') else str(job.state)
        print(f"  Status: {state}")

        if state in ("JOB_STATE_SUCCEEDED", "SUCCEEDED", "4"):
            print("Job completed!")
            return True
        elif state in ("JOB_STATE_FAILED", "FAILED", "5"):
            print(f"Job failed: {job.error}")
            return False
        elif state in ("JOB_STATE_CANCELLED", "CANCELLED", "6"):
            print("Job cancelled")
            return False

        time.sleep(60)  # check every minute

# ═════════════════════════════════════════
# STEP 5: DOWNLOAD + SAVE RESULTS
# ═════════════════════════════════════════

def download_and_save_results(mapping):
    """Download batch output from GCS and save as chunk files"""

    bucket = storage_client.bucket(BUCKET_NAME)
    output_blobs = list(bucket.list_blobs(prefix="batch_output/"))

    # Find the output JSONL file
    jsonl_blobs = [b for b in output_blobs if b.name.endswith(".jsonl")]

    if not jsonl_blobs:
        print("No output files found!")
        return

    print(f"Found {len(jsonl_blobs)} output files")

    all_results = []
    for blob in jsonl_blobs:
        content = blob.download_as_text()
        for line in content.strip().split("\n"):
            if line.strip():
                all_results.append(json.loads(line))

    print(f"Total results: {len(all_results)}")

    saved = 0
    errors = 0

    for i, result in enumerate(all_results):
        if i >= len(mapping):
            break

        info = mapping[i]
        merge_code = info["merge_code"]
        module_id = info["module_id"]
        chunk_type = info["chunk_type"]
        chunk_index = info["chunk_index"]
        seq = info["seq"]

        chunk_dir = os.path.join(CHUNKS_FOLDER, merge_code, module_id)
        os.makedirs(chunk_dir, exist_ok=True)

        try:
            # Extract response text
            response = result.get("response", {})
            candidates = response.get("candidates", [])
            if not candidates:
                errors += 1
                continue

            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            text = text.strip()

            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            chunks = json.loads(text)
            if isinstance(chunks, dict):
                chunks = [chunks]

            # Save each slide
            for slide_idx, slide in enumerate(chunks):
                slide["chunk_id"] = f"{module_id}_chunk_{seq:03d}"
                if slide_idx == 0:
                    fname = f"{seq:03d}_{chunk_type}_{chunk_index}.json"
                else:
                    fname = f"{seq:03d}_{chunk_type}_cont_{chunk_index}_{slide_idx}.json"

                with open(os.path.join(chunk_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(slide, f, indent=2, ensure_ascii=False)

                seq += 1

            saved += 1

        except Exception as e:
            print(f"  Error processing {info['label']}: {e}")
            errors += 1

    print(f"\nSaved: {saved}, Errors: {errors}")

    # Now create _meta.json for each module
    create_meta_files()

def create_meta_files():
    """Create _meta.json for modules that don't have one"""
    for mc in sorted(os.listdir(CHUNKS_FOLDER)):
        mc_path = os.path.join(CHUNKS_FOLDER, mc)
        if not os.path.isdir(mc_path):
            continue
        for mod_dir in sorted(os.listdir(mc_path)):
            mod_path = os.path.join(mc_path, mod_dir)
            if not os.path.isdir(mod_path):
                continue

            meta_path = os.path.join(mod_path, "_meta.json")

            chunk_files = sorted([
                f for f in os.listdir(mod_path)
                if f.endswith(".json") and f != "_meta.json"
            ])

            if not chunk_files:
                continue

            # Load module info
            module_path = None
            for mmc in os.listdir(MODULES_FOLDER):
                mp = os.path.join(MODULES_FOLDER, mmc, f"{mod_dir}.json")
                if os.path.exists(mp):
                    module_path = mp
                    break

            module_data = {}
            if module_path:
                with open(module_path, "r", encoding="utf-8") as f:
                    module_data = json.load(f)

            meta = {
                "module_id": mod_dir,
                "module_title": module_data.get("module_title", ""),
                "chapter_title": module_data.get("chapter_title", ""),
                "class": module_data.get("class", ""),
                "chapter": module_data.get("chapter", ""),
                "domain": module_data.get("domain", ""),
                "target_exam": "GPSTR",
                "total_slides": len(chunk_files),
                "chunk_order": chunk_files,
                "errors": []
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Meta files created/updated")

# ═════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════

def main():
    print(f"{'='*60}")
    print(f"BATCH CHUNK GENERATION")
    print(f"{'='*60}")

    # Step 1: Find pending calls
    print("\nStep 1: Finding pending API calls...")
    pending = get_pending_calls()
    print(f"Found {len(pending)} pending calls")

    if not pending:
        print("Nothing to do!")
        return

    # Step 2: Create JSONL and upload
    print("\nStep 2: Creating JSONL and uploading to GCS...")
    input_uri, mapping = create_and_upload_jsonl(pending)

    # Step 3: Submit batch job
    print("\nStep 3: Submitting batch job...")
    job = submit_batch_job(input_uri)

    # Step 4: Wait for completion
    success = wait_for_job(job)

    if not success:
        print("Batch job failed. Check Cloud Console for details.")
        return

    # Step 5: Download and save
    print("\nStep 5: Downloading results and saving chunks...")
    download_and_save_results(mapping)

    print("\nDone! Run check_status.py to see progress.")

if __name__ == "__main__":
    main()
