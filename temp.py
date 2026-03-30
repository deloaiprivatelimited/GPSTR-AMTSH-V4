import os
import json

VALIDATION_FOLDER = "validation_results"


def main():

    total_chapters = 0
    ready_chapters = 0

    total_modules = 0
    ready_modules = 0

    for file in os.listdir(VALIDATION_FOLDER):

        if not file.endswith("_validation.json"):
            continue

        path = os.path.join(VALIDATION_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_chapters += 1

        if data["chapter_recommendation"] == "READY_TO_GO":
            ready_chapters += 1

        modules = data.get("module_results", [])

        total_modules += len(modules)

        for m in modules:
            if m["release_recommendation"] == "READY_TO_GO":
                ready_modules += 1

    print("\n──────── VALIDATION SUMMARY ────────")
    print(f"Total Chapters       : {total_chapters}")
    print(f"Chapters READY_TO_GO : {ready_chapters}")
    print()
    print(f"Total Modules        : {total_modules}")
    print(f"Modules READY_TO_GO  : {ready_modules}")
    print("────────────────────────────────────")


if __name__ == "__main__":
    main()