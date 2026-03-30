# import os
# import json
# import shutil

# VALIDATION_FOLDER = "validation_results"
# MODULES_FOLDER = "modules"

# deleted = []
# kept = []

# for file in os.listdir(VALIDATION_FOLDER):

#     if not file.endswith("_validation.json"):
#         continue

#     report_path = os.path.join(VALIDATION_FOLDER, file)

#     with open(report_path, "r", encoding="utf-8") as f:
#         report = json.load(f)

#     chapter = report["chapter_name"]
#     recommendation = report["chapter_recommendation"]

#     chapter_path = os.path.join(MODULES_FOLDER, chapter)

#     if recommendation == "READY_TO_GO":
#         kept.append(chapter)

#     else:
#         # delete chapter folder
#         if os.path.exists(chapter_path):
#             shutil.rmtree(chapter_path)

#         # delete validation file
#         if os.path.exists(report_path):
#             os.remove(report_path)

#         deleted.append(chapter)

# print("\nCleanup finished")
# print("Deleted chapters:", deleted)
# print("Kept chapters:", kept)