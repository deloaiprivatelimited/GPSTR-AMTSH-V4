import os
import json
import shutil

MODULES_FOLDER = "modules"
VALIDATION_FOLDER = "validation_results"

deleted_chapters = []

for file in os.listdir(VALIDATION_FOLDER):

    if not file.endswith("_validation.json"):
        continue

    validation_path = os.path.join(VALIDATION_FOLDER, file)

    with open(validation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chapter = data["chapter_name"]

    if data["chapter_recommendation"] != "READY_TO_GO":

        module_path = os.path.join(MODULES_FOLDER, chapter)

        # delete module folder
        if os.path.exists(module_path):
            shutil.rmtree(module_path)

        # delete validation file
        os.remove(validation_path)

        deleted_chapters.append(chapter)

print("Deleted chapters:", len(deleted_chapters))
for c in deleted_chapters:
    print("-", c)