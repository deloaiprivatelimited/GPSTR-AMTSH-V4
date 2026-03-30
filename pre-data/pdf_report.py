import os
from PyPDF2 import PdfReader

total_pages = 0
page_counts = {}

for root, dirs, files in os.walk("./merged"):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            try:
                reader = PdfReader(pdf_path)
                pages = len(reader.pages)
                page_counts[pdf_path] = pages
                total_pages += pages
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")

num_pdfs = len(page_counts)

if num_pdfs == 0:
    print("No PDFs found.")
    exit()

avg_pages = total_pages / num_pdfs
max_pdf = max(page_counts, key=page_counts.get)
min_pdf = min(page_counts, key=page_counts.get)

print("PDF REPORT")
print("-----------")
print(f"Total PDFs: {num_pdfs}")
print(f"Total Pages: {total_pages}")
print(f"Average Pages per PDF: {avg_pages:.2f}")
print(f"Max Pages: {page_counts[max_pdf]} ({max_pdf})")
print(f"Min Pages: {page_counts[min_pdf]} ({min_pdf})")