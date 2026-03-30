# import json
# import os
# # from PyPDF2 import PdfMerger

# DATA_FILE = "data.json"
# TEXTBOOK_DIR = "textbooks"
# OUTPUT_DIR = "merged"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# with open(DATA_FILE) as f:
#     data = json.load(f)

# merge_groups = {}

# # Collect PDFs per merge_code
# for subject in data:
#     for chapter in data[subject]:

#         code = chapter["merge_code"]
#         cls = str(chapter["class"])
#         ch_no = chapter["chapter_no"]

#         pdf_path = os.path.join(
#             TEXTBOOK_DIR,
#             cls,
#             f"Chapter_{ch_no:02d}.pdf"
#         )

#         merge_groups.setdefault(code, []).append(pdf_path)

# # Merge PDFs
# for code, pdfs in merge_groups.items():

#     # merger = PdfMerger()

#     for pdf in sorted(pdfs):
#         if os.path.exists(pdf):
#             merger.append(pdf)
#         else:
#             print("Missing:", pdf)

#     output = os.path.join(OUTPUT_DIR, f"{code}.pdf")
#     merger.write(output)
#     merger.close()

#     print("Created:", output)