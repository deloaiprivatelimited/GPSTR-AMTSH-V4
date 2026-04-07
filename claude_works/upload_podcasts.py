"""
upload_podcasts.py
-------------------
Uploads podcast_final.wav files to S3 (concept-wise) and saves metadata to MongoDB.
Reads podcast JSON for module titles and metadata.
Does NOT delete local files.

S3 structure: podcasts/{merge_code}/{module_id}.wav
MongoDB collection: podcasts
"""

import json
import wave
import os
import sys
import time
import boto3
from pathlib import Path
from datetime import datetime, timezone
from pymongo import MongoClient

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Load .env ──
def load_dotenv(path=".env"):
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_dotenv()

# ── Config ──
PODCAST_AUDIO_DIR = Path("claude_works/podcast_audio")
PODCAST_JSON_DIR  = Path("claude_works/podcasts")
DATA_JSON         = Path("data.json")

AWS_ACCESS_KEY  = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET       = os.environ.get("AWS_S3_BUCKET", "azad")

MONGODB_URI     = os.environ.get("MONGODB_URI", "mongodb+srv://user:user@cluster0.rgocxdb.mongodb.net/gpstr-maths-db")
MONGO_DB_NAME   = "gpstr-maths-db"
MONGO_COLLECTION = "podcasts"

MAX_RETRIES = 3


# ── Load chapter metadata from data.json ──
def load_chapter_meta():
    if not DATA_JSON.exists():
        return {}
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for domain, chapters in data.items():
        for ch in chapters:
            mc = ch["merge_code"]
            if mc not in lookup:
                lookup[mc] = {
                    "domain": domain,
                    "chapter_name": ch["chapter_name"],
                    "classes": [],
                    "chapters": [],
                }
            lookup[mc]["classes"].append(ch["class"])
            lookup[mc]["chapters"].append(ch["chapter_no"])
    for v in lookup.values():
        v["classes"] = sorted(set(v["classes"]))
        v["chapters"] = sorted(set(v["chapters"]))
    return lookup

CHAPTER_META = load_chapter_meta()


def get_wav_duration(wav_path):
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


def upload_to_s3(local_path, s3_key):
    s3 = get_s3_client()
    s3.upload_file(
        str(local_path), S3_BUCKET, s3_key,
        ExtraArgs={"ContentType": "audio/wav"},
    )
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


def upload_with_retry(local_path, s3_key):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return upload_to_s3(local_path, s3_key)
        except Exception as e:
            print(f"    S3 FAIL (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
            else:
                raise


def get_podcast_meta(merge_code, module_id):
    """Read podcast JSON for module title and metadata."""
    json_path = PODCAST_JSON_DIR / merge_code / f"{module_id}_podcast.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Podcast Upload -> S3 + MongoDB\n")

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        print("FATAL: AWS credentials not set in .env")
        return

    client = MongoClient(MONGODB_URI)
    db = client[MONGO_DB_NAME]
    col = db[MONGO_COLLECTION]

    # Check already uploaded
    existing = set(
        doc["merge_code"] for doc in col.find({}, {"merge_code": 1})
    )
    print(f"Already in MongoDB: {len(existing)} chapters\n")

    # Collect all chapters
    if not PODCAST_AUDIO_DIR.is_dir():
        print(f"No podcast audio dir: {PODCAST_AUDIO_DIR}")
        return

    chapters = sorted([d.name for d in PODCAST_AUDIO_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(chapters)} chapters with podcast audio\n")

    total_uploaded = 0
    total_skipped = 0

    for idx, merge_code in enumerate(chapters, 1):
        if merge_code in existing:
            print(f"[{idx}/{len(chapters)}] SKIP {merge_code} (already in MongoDB)")
            total_skipped += 1
            continue

        ch_dir = PODCAST_AUDIO_DIR / merge_code
        concept_dirs = sorted([d for d in ch_dir.iterdir() if d.is_dir()])

        if not concept_dirs:
            continue

        print(f"[{idx}/{len(chapters)}] {merge_code} — {len(concept_dirs)} concepts")

        ch_meta = CHAPTER_META.get(merge_code, {})
        modules = []

        for concept_dir in concept_dirs:
            module_id = concept_dir.name
            final_wav = concept_dir / "podcast_final.wav"

            if not final_wav.exists():
                print(f"  SKIP {module_id}: no podcast_final.wav")
                continue

            duration = get_wav_duration(final_wav)
            size_mb = final_wav.stat().st_size / 1_000_000
            s3_key = f"podcasts/{merge_code}/{module_id}.wav"

            # Get podcast metadata
            pod_meta = get_podcast_meta(merge_code, module_id)
            module_title = pod_meta.get("module_title", "")
            podcast_info = pod_meta.get("podcast_meta", {})

            print(f"  Uploading {module_id} ({size_mb:.1f} MB, {duration/60:.1f} min)...")
            try:
                s3_url = upload_with_retry(final_wav, s3_key)
            except Exception as e:
                print(f"  FAILED {module_id}: {e}")
                continue

            modules.append({
                "module_id": module_id,
                "module_title": module_title,
                "duration_seconds": round(duration, 2),
                "duration_display": f"{int(duration // 60)}m {int(duration % 60)}s",
                "file_size_mb": round(size_mb, 2),
                "s3_url": s3_url,
                "s3_key": s3_key,
                "total_dialogues": podcast_info.get("total_dialogues", 0),
                "estimated_duration_minutes": podcast_info.get("estimated_duration_minutes", 0),
                "sections_covered": podcast_info.get("sections_covered", []),
            })

        if not modules:
            continue

        total_dur = sum(m["duration_seconds"] for m in modules)

        doc = {
            "merge_code": merge_code,
            "chapter_name": ch_meta.get("chapter_name", ""),
            "domain": ch_meta.get("domain", ""),
            "classes": ch_meta.get("classes", []),
            "chapter_numbers": ch_meta.get("chapters", []),
            "modules": modules,
            "total_concepts": len(modules),
            "total_duration_seconds": round(total_dur, 2),
            "total_duration_display": f"{int(total_dur // 60)}m {int(total_dur % 60)}s",
            "s3_bucket": S3_BUCKET,
            "status": "published",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        col.update_one(
            {"merge_code": merge_code},
            {"$set": doc},
            upsert=True,
        )
        print(f"  MongoDB OK — {len(modules)} concepts, {total_dur/60:.1f} min total")
        total_uploaded += 1

    print(f"\nDone! Uploaded: {total_uploaded}, Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
