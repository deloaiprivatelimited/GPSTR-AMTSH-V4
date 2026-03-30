"""
estimate_tokens.py — Token & Cost Estimator for generate_chunks.py
Reads validation_results/ → finds READY_TO_GO modules → estimates tokens + cost.
No API calls made. Purely local calculation.
"""

import os
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

MODULES_DIR      = Path("modules")
VALIDATION_DIR   = Path("validation_results")
PROMPT_PATH      = Path("prompts/generate_chunks.txt")

# gpt-4.1 pricing (per 1M tokens) — update if prices change
INPUT_COST_PER_1M  = 2.00   # USD
OUTPUT_COST_PER_1M = 8.00   # USD

# Rough output multiplier:
# Each module JSON produces ~3x tokens worth of chunk JSON output
# Adjust this if your actual outputs are larger/smaller
OUTPUT_MULTIPLIER = 3.0

# ============================================================
# SIMPLE TOKENIZER  (no tiktoken needed)
# ~4 characters per token — standard GPT estimate
# ============================================================

def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ============================================================
# READ VALIDATION → READY_TO_GO MODULE IDS
# ============================================================

def get_ready_module_ids() -> set:
    ready_ids = set()
    if not VALIDATION_DIR.exists():
        print(f"[WARN] Validation folder not found: {VALIDATION_DIR}")
        return ready_ids
    for vfile in VALIDATION_DIR.glob("*_validation.json"):
        try:
            with open(vfile, "r", encoding="utf-8") as f:
                report = json.load(f)
            for m in report.get("module_results", []):
                if m.get("release_recommendation") == "READY_TO_GO" or True:
                    ready_ids.add(m["module_id"])
        except Exception as e:
            print(f"[WARN] Could not read {vfile.name}: {e}")
    return ready_ids


# ============================================================
# LOAD PROMPT
# ============================================================

def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        print(f"[WARN] Prompt not found at {PROMPT_PATH} — using empty string")
        return ""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# MAIN ESTIMATE
# ============================================================

def main():

    print("=" * 60)
    print("  TOKEN & COST ESTIMATOR — generate_chunks.py")
    print("=" * 60)

    # 1. Load prompt
    prompt_text   = load_prompt()
    prompt_tokens = count_tokens(prompt_text)
    print(f"\nPrompt template : {prompt_tokens:,} tokens  ({len(prompt_text):,} chars)")

    # 2. Find READY_TO_GO modules
    ready_ids = get_ready_module_ids()
    print(f"READY_TO_GO     : {len(ready_ids)} module(s)\n")

    if not ready_ids:
        print("Nothing to estimate. Run validate_modules.py first.")
        return

    # 3. Walk module files
    module_files = sorted(MODULES_DIR.rglob("*.json"))

    rows = []
    skipped = 0

    for path in module_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        module_id = data.get("module_id", path.stem)

        if module_id not in ready_ids:
            skipped += 1
            continue

        module_text   = json.dumps(data, ensure_ascii=False)
        module_tokens = count_tokens(module_text)

        # Input = prompt + module JSON
        input_tokens  = prompt_tokens + module_tokens

        # Output = estimated chunk JSON (3x module size is typical)
        output_tokens = int(module_tokens * OUTPUT_MULTIPLIER)

        rows.append({
            "module_id":      module_id,
            "module_tokens":  module_tokens,
            "input_tokens":   input_tokens,
            "output_tokens":  output_tokens,
        })

    if not rows:
        print("No matching module files found in modules/")
        return

    # 4. Totals
    total_input  = sum(r["input_tokens"]  for r in rows)
    total_output = sum(r["output_tokens"] for r in rows)
    total_tokens = total_input + total_output

    cost_input  = (total_input  / 1_000_000) * INPUT_COST_PER_1M
    cost_output = (total_output / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost  = cost_input + cost_output

    # 5. Per-module table
    print(f"{'MODULE ID':<45} {'INPUT':>8} {'OUTPUT':>8} {'TOTAL':>8}")
    print("-" * 75)
    for r in rows:
        total_row = r["input_tokens"] + r["output_tokens"]
        print(
            f"{r['module_id']:<45} "
            f"{r['input_tokens']:>8,} "
            f"{r['output_tokens']:>8,} "
            f"{total_row:>8,}"
        )

    # 6. Summary
    print("=" * 75)
    print(f"{'Modules to process':<35} {len(rows)}")
    print(f"{'Modules skipped (not READY)':<35} {skipped}")
    print()
    print(f"{'Total INPUT tokens':<35} {total_input:>12,}")
    print(f"{'Total OUTPUT tokens (est.)':<35} {total_output:>12,}")
    print(f"{'Total tokens':<35} {total_tokens:>12,}")
    print()
    print(f"{'Input cost  (@ ${INPUT_COST_PER_1M}/1M)':<35} ${cost_input:>10.4f}")
    print(f"{'Output cost (@ ${OUTPUT_COST_PER_1M}/1M)':<35} ${cost_output:>10.4f}")
    print(f"{'ESTIMATED TOTAL COST':<35} ${total_cost:>10.4f}")
    print("=" * 75)
    print()
    print("Notes:")
    print(f"  • Token counts use ~4 chars/token estimate (no tiktoken)")
    print(f"  • Output estimate uses {OUTPUT_MULTIPLIER}x input module size")
    print(f"  • Actual output varies by module complexity (2x–4x range)")
    print(f"  • Prices: gpt-4.1 input=${INPUT_COST_PER_1M}/1M  output=${OUTPUT_COST_PER_1M}/1M")
    print(f"  • Update INPUT_COST_PER_1M / OUTPUT_COST_PER_1M at top of file if needed")
    print()


if __name__ == "__main__":
    main()