#!/bin/bash
# ─────────────────────────────────────────────
# STAGE 1 — Extract text from PDFs using Claude CLI
# Runs MAX_PARALLEL jobs concurrently, each with fresh context
# ─────────────────────────────────────────────

PDF_DIR="merged"
OUTPUT_DIR="claude_works/extracted_text"
PROMPT_FILE="prompts_v2/1_extract_text.txt"
MAX_PARALLEL=5

mkdir -p "$OUTPUT_DIR"

# counters
total=0
skipped=0
started=0
failed=0

# track background pids
pids=()

process_pdf() {
  local pdf_file="$1"
  local name=$(basename "$pdf_file" .pdf)
  local output_file="$OUTPUT_DIR/${name}.txt"
  local abs_pdf
  abs_pdf="$(cd "$(dirname "$pdf_file")" && pwd)/$(basename "$pdf_file")"
  local prompt
  prompt=$(cat "$PROMPT_FILE")

  # --allowed-tools Read lets Claude read the PDF visually (important for Kannada fonts)
  # Prompt piped via stdin, stdout captured to output file
  # For large PDFs (>20 pages), Claude must read in batches of 20 pages
  echo "Read the file ${abs_pdf} — this is a Kannada medium maths textbook PDF.
IMPORTANT: The Read tool can only read 20 pages at a time. If the PDF has more than 20 pages, you MUST read it in batches using the 'pages' parameter (e.g., pages: \"1-20\", then \"21-40\", etc.) until you have read ALL pages. Do NOT skip any pages.
After reading ALL pages, apply these instructions and output ONLY the extracted text, nothing else. No commentary, no explanations — just the extracted content.

${prompt}" | claude -p --allowed-tools "Read" > "$output_file" 2>/dev/null

  if [ -s "$output_file" ]; then
    echo "  ✓ Done: $name ($(wc -l < "$output_file") lines)"
  else
    rm -f "$output_file"
    echo "  ✗ Failed: $name"
  fi
}

echo "═══════════════════════════════════════"
echo " PDF Text Extraction — Claude CLI"
echo "═══════════════════════════════════════"
echo " Source:  $PDF_DIR/"
echo " Output:  $OUTPUT_DIR/"
echo " Workers: $MAX_PARALLEL"
echo "═══════════════════════════════════════"
echo ""

for pdf_file in "$PDF_DIR"/*.pdf; do
  name=$(basename "$pdf_file" .pdf)
  output_file="$OUTPUT_DIR/${name}.txt"
  total=$((total + 1))

  # skip if already done
  if [ -f "$output_file" ] && [ -s "$output_file" ]; then
    echo "  ⏭ Skip: $name (already exists)"
    skipped=$((skipped + 1))
    continue
  fi

  # wait if we hit max parallel limit
  while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
    # wait for any one to finish
    wait -n 2>/dev/null || true
    # clean up finished pids
    new_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      fi
    done
    pids=("${new_pids[@]}")
  done

  echo "  🚀 Starting: $name"
  process_pdf "$pdf_file" &
  pids+=($!)
  started=$((started + 1))
done

# wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to finish..."
wait

echo ""
echo "═══════════════════════════════════════"
echo " DONE"
echo " Total PDFs:  $total"
echo " Skipped:     $skipped"
echo " Processed:   $started"
echo "═══════════════════════════════════════"
