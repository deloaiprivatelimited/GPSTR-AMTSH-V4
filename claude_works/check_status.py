"""
Pipeline Status Checker — run anytime to see progress
python claude_works/check_status.py
"""
import os
import json

MODULES_FOLDER = "claude_works/modules"
CHUNKS_FOLDER = "claude_works/chunks"
VALIDATION_FOLDER = "claude_works/chunk_validation"
AUDIO_FOLDER = "claude_works/audio"

def count_modules():
    total = 0
    total_api_calls = 0
    for mc in os.listdir(MODULES_FOLDER):
        mp = os.path.join(MODULES_FOLDER, mc)
        if not os.path.isdir(mp): continue
        for f in os.listdir(mp):
            if not f.endswith(".json"): continue
            total += 1
            try:
                d = json.load(open(os.path.join(mp, f), encoding="utf-8"))
                t = d.get("theory", {})
                defs = len(t.get("definitions", []))
                formulas = len(t.get("formulas", []))
                theorems = len(t.get("theorems", []))
                props = len(t.get("properties", []))
                examples = len(d.get("worked_examples", []))
                prop_batches = (props + 2) // 3 if props else 0
                # 1 intro + defs + theorems + prop_batches + formulas + examples + 1 recap
                total_api_calls += 1 + defs + theorems + prop_batches + formulas + examples + 1
            except:
                pass
    return total, total_api_calls

def check_chunks():
    done = 0
    errors = 0
    total_slides = 0
    total_words = 0
    completed_api_calls = 0
    failed_api_calls = 0

    if not os.path.isdir(CHUNKS_FOLDER):
        return 0, 0, 0, 0, 0, 0

    for mc in os.listdir(CHUNKS_FOLDER):
        mp = os.path.join(CHUNKS_FOLDER, mc)
        if not os.path.isdir(mp): continue
        for mod in os.listdir(mp):
            mod_dir = os.path.join(mp, mod)
            if not os.path.isdir(mod_dir): continue
            meta = os.path.join(mod_dir, "_meta.json")
            if not os.path.exists(meta): continue
            d = json.load(open(meta, encoding="utf-8"))
            total_slides += d.get("total_slides", 0)
            errs = d.get("errors", [])
            failed_api_calls += len(errs)
            # Count chunk files = successful API calls (each file = 1 slide from an API call)
            chunk_files = [f for f in os.listdir(mod_dir) if f.endswith(".json") and f != "_meta.json"]
            # Group by API call (count unique prefixes, not cont files)
            api_groups = set()
            for cf in chunk_files:
                parts = cf.split("_", 1)
                if len(parts) >= 2 and "_cont_" not in parts[1]:
                    api_groups.add(cf)
            completed_api_calls += len(api_groups)

            if errs:
                errors += 1
            else:
                done += 1

            for f in chunk_files:
                try:
                    c = json.load(open(os.path.join(mod_dir, f), encoding="utf-8"))
                    script = c.get("script", "")
                    if script:
                        total_words += len(script.split())
                except:
                    pass

    return done, errors, total_slides, total_words, completed_api_calls, failed_api_calls

def check_validation():
    done = 0
    ready = 0
    not_ready = 0

    if not os.path.isdir(VALIDATION_FOLDER):
        return 0, 0, 0

    for mc in os.listdir(VALIDATION_FOLDER):
        mp = os.path.join(VALIDATION_FOLDER, mc)
        if not os.path.isdir(mp): continue
        for mod in os.listdir(mp):
            marker = os.path.join(mp, mod, "_done.marker")
            if not os.path.exists(marker): continue
            done += 1
            with open(marker) as f:
                content = f.read()
            if "not_ready: 0" in content:
                ready += 1
            else:
                not_ready += 1

    return done, ready, not_ready

def check_audio():
    done = 0
    total_duration = 0.0
    total_chunks = 0

    if not os.path.isdir(AUDIO_FOLDER):
        return 0, 0, 0

    for mc in os.listdir(AUDIO_FOLDER):
        mp = os.path.join(AUDIO_FOLDER, mc)
        if not os.path.isdir(mp): continue
        for mod in os.listdir(mp):
            tl_path = os.path.join(mp, mod, "timeline.json")
            if not os.path.exists(tl_path): continue
            try:
                tl = json.load(open(tl_path, encoding="utf-8"))
                done += 1
                total_duration += tl.get("total_duration", 0)
                total_chunks += tl.get("total_chunks", 0)
            except:
                pass

    return done, total_duration, total_chunks

def main():
    total_modules, total_api_needed = count_modules()
    chunks_done, chunks_errors, total_slides, total_words, api_done, api_failed = check_chunks()
    val_done, val_ready, val_not_ready = check_validation()
    audio_done, audio_duration, audio_chunks = check_audio()

    print("=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    # Modules
    print(f"\n1. MODULES: {total_modules} total")
    print(f"   Total API calls needed: {total_api_needed}")

    # Chunks
    chunks_total = chunks_done + chunks_errors
    chunks_pct = chunks_total * 100 // max(total_modules, 1)
    print(f"\n2. CHUNK GENERATION: {chunks_total}/{total_modules} modules ({chunks_pct}%)")
    print(f"   Clean: {chunks_done} | With errors: {chunks_errors}")
    print(f"   API calls: {api_done} done / {api_failed} failed / {total_api_needed - api_done - api_failed} pending")
    print(f"   Total slides: {total_slides}")
    print(f"   Total script words: {total_words:,}")
    if chunks_done > 0:
        print(f"   Avg slides/module: {total_slides / chunks_done:.1f}")
        print(f"   Avg words/slide: {total_words / max(total_slides, 1):.0f}")

    # Estimated total
    if chunks_done > 0:
        ratio = total_modules / chunks_done
        est_slides = int(total_slides * ratio)
        est_words = int(total_words * ratio)
        est_hours = est_words / 2.5 / 3600
        print(f"\n   --- Estimated all {total_modules} modules ---")
        print(f"   Slides: ~{est_slides:,}")
        print(f"   Words: ~{est_words:,}")
        print(f"   Audio: ~{est_hours:.0f} hours")

    # Validation
    val_pct = val_done * 100 // max(chunks_done, 1) if chunks_done else 0
    print(f"\n3. VALIDATION: {val_done}/{chunks_done} ({val_pct}%)")
    print(f"   Production ready: {val_ready}")
    print(f"   Not ready: {val_not_ready}")

    # Audio
    audio_pct = audio_done * 100 // max(val_ready, 1) if val_ready else 0
    audio_hrs = audio_duration / 3600
    audio_mins = (audio_duration % 3600) / 60
    print(f"\n4. AUDIO: {audio_done}/{val_ready} ({audio_pct}%)")
    print(f"   Duration: {int(audio_hrs)}h {int(audio_mins)}m")
    print(f"   Chunks with audio: {audio_chunks}")

    # Overall
    print(f"\n{'=' * 60}")
    print(f"OVERALL: ", end="")
    if audio_done == total_modules:
        print("ALL DONE!")
    elif val_ready == total_modules:
        print("Chunks + Validation done. Audio in progress.")
    elif chunks_done == total_modules:
        print("Chunks done. Validation in progress.")
    else:
        print("Chunk generation in progress.")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
