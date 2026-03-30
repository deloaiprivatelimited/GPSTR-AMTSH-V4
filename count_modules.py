import os, json

VALIDATION_FOLDER = "validation_results"

ready_chapters = 0
blocked_chapters = 0

for f in os.listdir(VALIDATION_FOLDER):
    if not f.endswith("_validation.json"):
        continue

    with open(os.path.join(VALIDATION_FOLDER, f), "r", encoding="utf-8") as file:
        data = json.load(file)

    if data["chapter_recommendation"] == "READY_TO_GO":
        ready_chapters += 1
    else:
        blocked_chapters += 1

print("READY CHAPTERS:", ready_chapters)
print("BLOCKED CHAPTERS:", blocked_chapters)
print("TOTAL CHAPTERS  :", ready_chapters + blocked_chapters)