import os
import io
import pypdf
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "project-6565cf16-a3d4-4f6e-935"
LOCATION = "us-central1"

PDF_PATH = "Medieval history.pdf"
OUTPUT_FOLDER = "claude_works/extracted_kannada"
PAGES_TO_PROCESS = 10  # first 10 pages

# -----------------------------
# INIT
# -----------------------------
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# EXTRACT SINGLE PAGE AS PDF BYTES
# -----------------------------
def extract_page_as_pdf(pdf_path, page_num):
    """Extract a single page from PDF and return as PDF bytes."""
    reader = pypdf.PdfReader(pdf_path)
    writer = pypdf.PdfWriter()
    writer.add_page(reader.pages[page_num])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

# -----------------------------
# STEP 1: EXTRACT KANNADA TEXT (one page at a time)
# -----------------------------
def extract_kannada_from_page(pdf_bytes, page_num):
    """Send one page to Gemini, get back exact Kannada text."""
    pdf_part = Part.from_data(data=pdf_bytes, mime_type="application/pdf")

    prompt = f"""This is page {page_num + 1} of a handwritten Kannada manuscript on medieval history.

Your task: Extract the EXACT Kannada text written on this page.

Rules:
- Output ONLY the Kannada text, nothing else
- Preserve the original Kannada script exactly as written
- Do NOT translate to English
- Do NOT add any commentary, headers, or labels
- If a word is unclear, do your best to reproduce it
- Maintain paragraph structure with blank lines"""

    response = model.generate_content(
        [pdf_part, prompt],
        generation_config={"temperature": 0.1}
    )
    return response.text

# -----------------------------
# STEP 2: TRANSLATE KANNADA → ENGLISH
# -----------------------------
def translate_to_english(kannada_text, page_num):
    """Send one page's Kannada text to Gemini for English translation."""
    prompt = f"""Below is Kannada text extracted from page {page_num + 1} of a handwritten manuscript on medieval history.

Translate it into clear, accurate English. Preserve paragraph structure.
Do NOT include the original Kannada — output ONLY the English translation.

---
{kannada_text}
---

English translation:"""

    response = model.generate_content(
        [prompt],
        generation_config={"temperature": 0.2}
    )
    return response.text

# -----------------------------
# MAIN
# -----------------------------
def main():
    kannada_output_path = os.path.join(OUTPUT_FOLDER, "kannada_pages_1_to_10.txt")
    english_output_path = os.path.join(OUTPUT_FOLDER, "english_pages_1_to_10.txt")

    all_kannada = []
    all_english = []

    print(f"📄 Processing first {PAGES_TO_PROCESS} pages of: {PDF_PATH}\n")

    for page_num in range(PAGES_TO_PROCESS):
        print(f"  [{page_num + 1}/{PAGES_TO_PROCESS}] Extracting Kannada...", end=" ", flush=True)

        try:
            pdf_bytes = extract_page_as_pdf(PDF_PATH, page_num)
            kannada_text = extract_kannada_from_page(pdf_bytes, page_num)
            all_kannada.append(f"--- Page {page_num + 1} ---\n{kannada_text}")
            print("✅  Translating...", end=" ", flush=True)
        except Exception as e:
            all_kannada.append(f"--- Page {page_num + 1} ---\n[ERROR: {e}]")
            all_english.append(f"--- Page {page_num + 1} ---\n[EXTRACTION ERROR: {e}]")
            print(f"❌ {e}")
            continue

        try:
            english_text = translate_to_english(kannada_text, page_num)
            all_english.append(f"--- Page {page_num + 1} ---\n{english_text}")
            print("✅")
        except Exception as e:
            all_english.append(f"--- Page {page_num + 1} ---\n[TRANSLATION ERROR: {e}]")
            print(f"❌ translate failed: {e}")

    # Save combined Kannada text
    with open(kannada_output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_kannada))
    print(f"\n✅ Kannada text saved: {kannada_output_path}")

    # Save combined English text
    with open(english_output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_english))
    print(f"✅ English translation saved: {english_output_path}")

    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
