import csv
import os
import re
from PyPDF2 import PdfReader, PdfWriter

CSV_PATH = "science_page_numbers.csv"
PDF_DIR = "science_textbooks/Science"
OUTPUT_DIR = "science_textbooks/split"

def normalize(name):
    """Collapse spaces and normalize hyphens for matching."""
    name = re.sub(r'\s+', '', name.strip().lower())  # remove all spaces
    return name

def find_pdf(source_name, pdf_files):
    """Find the actual PDF file matching the CSV source_pdf value."""
    norm_source = normalize(source_name)
    for f in pdf_files:
        if normalize(f.replace('.pdf', '')) == norm_source:
            return f
    # Fallback: check if normalized source is contained in normalized filename
    for f in pdf_files:
        norm_f = normalize(f.replace('.pdf', ''))
        if norm_source in norm_f or norm_f in norm_source:
            return f
    return None

def extract_std(source_name):
    """Extract standard/class number from source PDF name."""
    match = re.match(r'(\d+)', source_name.strip())
    return match.group(1) if match else None

def main():
    pdf_files = os.listdir(PDF_DIR)

    # Cache loaded PDFs to avoid re-reading
    pdf_cache = {}

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chapter = int(row['chapter_number'])
            offset = int(row['offest'])  # typo in CSV header
            from_page = int(row['from_page'])
            to_page = int(row['to_page'])
            source_pdf = row['source_pdf']

            std = extract_std(source_pdf)
            if not std:
                print(f"Could not extract standard from: {source_pdf}")
                continue

            # Actual PDF page numbers (0-indexed for PyPDF2)
            start_page = from_page + offset - 1  # -1 for 0-indexed
            end_page = to_page + offset - 1       # inclusive

            # Find matching PDF
            matched_file = find_pdf(source_pdf, pdf_files)
            if not matched_file:
                print(f"PDF not found for: {source_pdf}")
                continue

            pdf_path = os.path.join(PDF_DIR, matched_file)

            # Load PDF (cached)
            if pdf_path not in pdf_cache:
                print(f"Loading: {matched_file}")
                pdf_cache[pdf_path] = PdfReader(pdf_path)

            reader_obj = pdf_cache[pdf_path]
            total_pages = len(reader_obj.pages)

            # Clamp end_page to total pages
            if end_page >= total_pages:
                print(f"  Warning: end_page {end_page+1} exceeds {total_pages} pages in {matched_file}, clamping")
                end_page = total_pages - 1

            if start_page >= total_pages:
                print(f"  Error: start_page {start_page+1} exceeds {total_pages} pages in {matched_file}, skipping")
                continue

            # Create output directory: split/std_6/
            out_dir = os.path.join(OUTPUT_DIR, f"std_{std}")
            os.makedirs(out_dir, exist_ok=True)

            # Extract pages
            writer = PdfWriter()
            for i in range(start_page, end_page + 1):
                writer.add_page(reader_obj.pages[i])

            out_file = os.path.join(out_dir, f"chapter_{chapter}.pdf")
            with open(out_file, 'wb') as out_f:
                writer.write(out_f)

            page_count = end_page - start_page + 1
            print(f"  Std {std} Chapter {chapter}: pages {start_page+1}-{end_page+1} ({page_count} pages) -> {out_file}")

    print("\nDone! All chapters split.")

if __name__ == "__main__":
    main()
