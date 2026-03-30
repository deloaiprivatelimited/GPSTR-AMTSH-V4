import os
import json
import csv

VALIDATION_FOLDER = "validation_results"
OUTPUT_FILE = "modules_needing_revision.csv"

rows = []

for file in os.listdir(VALIDATION_FOLDER):

    if not file.endswith("_validation.json"):
        continue

    path = os.path.join(VALIDATION_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    chapter = report["chapter_name"]

    for module in report["module_results"]:

        status = module.get("release_recommendation")

        # Only collect modules needing revision
        if status == "NEEDS_REVIEW":

            rows.append({
                "chapter": chapter,
                "module_id": module.get("module_id"),
                "major_issues": " | ".join(module.get("major_issues", [])),
                "minor_issues": " | ".join(module.get("minor_issues", []))
            })


# Write CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "chapter",
            "module_id",
            "major_issues",
            "minor_issues"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("CSV exported:", OUTPUT_FILE)
print("Modules needing revision:", len(rows))