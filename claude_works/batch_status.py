"""
Check batch job status + download results when done
Usage:
  python claude_works/batch_status.py          # just check status
  python claude_works/batch_status.py download  # download results after job succeeds
"""
import os
import json
import sys
import vertexai
from vertexai.batch_prediction import BatchPredictionJob

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"
BATCH_FOLDER = "claude_works/batch"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load job info
job_info_path = os.path.join(BATCH_FOLDER, "batch_job.json")
if not os.path.exists(job_info_path):
    print("No batch job found. Run batch_generate_chunks.py first.")
    sys.exit(1)

with open(job_info_path) as f:
    job_info = json.load(f)

job_name = job_info["job_name"]

# Get job status
job = BatchPredictionJob(job_name)
job.refresh()

state = job.state.name if hasattr(job.state, 'name') else str(job.state)

print(f"{'='*60}")
print(f"BATCH JOB STATUS")
print(f"{'='*60}")
print(f"Job: {job_name}")
print(f"Status: {state}")
print(f"Submitted: {job_info.get('submitted_at', '?')}")
print(f"Input: {job_info.get('input_uri', '?')}")
print(f"Output: {job_info.get('output_uri', '?')}")

# Load mapping for counts
mapping_path = os.path.join(BATCH_FOLDER, "batch_mapping.json")
if os.path.exists(mapping_path):
    mapping = json.load(open(mapping_path, encoding="utf-8"))
    print(f"Total prompts: {len(mapping)}")

if state in ("JOB_STATE_SUCCEEDED", "SUCCEEDED", "4"):
    print("\nJob COMPLETED! Run with 'download' to save results:")
    print("  python claude_works/batch_status.py download")

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        print("\nDownloading results...")
        from claude_works.batch_generate_chunks import download_and_save_results
        mapping = json.load(open(mapping_path, encoding="utf-8"))
        download_and_save_results(mapping)

elif state in ("JOB_STATE_FAILED", "FAILED", "5"):
    print(f"\nJob FAILED")
    if hasattr(job, 'error') and job.error:
        print(f"Error: {job.error}")

elif state in ("JOB_STATE_CANCELLED", "CANCELLED", "6"):
    print("\nJob CANCELLED")

else:
    print(f"\nJob still running. Check back later.")
    print(f"Console: https://console.cloud.google.com/ai/platform/locations/{LOCATION}/batch-predictions/{job_name.split('/')[-1]}?project={PROJECT_ID}")
