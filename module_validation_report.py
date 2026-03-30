import os
import json

VALIDATION_FOLDER = "validation_results"
MODULES_FOLDER = "modules"


def get_all_chapters():
    return sorted([
        d for d in os.listdir(MODULES_FOLDER)
        if os.path.isdir(os.path.join(MODULES_FOLDER, d))
    ])


def get_validation_reports():
    return [
        f for f in os.listdir(VALIDATION_FOLDER)
        if f.endswith("_validation.json")
    ]


def main():

    all_chapters = get_all_chapters()
    reports = get_validation_reports()

    validated_chapters = []
    ready_chapters = 0
    review_chapters = 0
    blocked_chapters = 0

    ready_modules = 0
    review_modules = 0
    blocked_modules = 0
    total_modules = 0

    for report_file in reports:

        path = os.path.join(VALIDATION_FOLDER, report_file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        validated_chapters.append(data["chapter_name"])

        total_modules += data["total_modules"]
        ready_modules += data["ready_to_go"]
        review_modules += data["needs_review"]
        blocked_modules += data["do_not_release"]

        if data["chapter_recommendation"] == "READY_TO_GO":
            ready_chapters += 1
        elif data["chapter_recommendation"] == "NEEDS_REVIEW":
            review_chapters += 1
        else:
            blocked_chapters += 1

    missing_chapters = [c for c in all_chapters if c not in validated_chapters]

    print("\n==============================")
    print("VALIDATION STATUS")
    print("==============================\n")

    print(f"Total Chapters Found        : {len(all_chapters)}")
    print(f"Chapters Validated          : {len(validated_chapters)}")
    print(f"Missing Validations         : {len(missing_chapters)}")

    print("\nChapter Status")
    print("------------------------------")

    print(f"READY_TO_GO chapters        : {ready_chapters}")
    print(f"NEEDS_REVIEW chapters       : {review_chapters}")
    print(f"DO_NOT_RELEASE chapters     : {blocked_chapters}")

    print("\nModule Status")
    print("------------------------------")

    print(f"Total modules validated     : {total_modules}")
    print(f"READY_TO_GO modules         : {ready_modules}")
    print(f"NEEDS_REVIEW modules        : {review_modules}")
    print(f"DO_NOT_RELEASE modules      : {blocked_modules}")

    print("\nSuccess Rate")
    print("------------------------------")

    if total_modules > 0:
        success_rate = (ready_modules / total_modules) * 100
        print(f"Module success rate         : {success_rate:.2f}%")

    print("\nMissing Chapter Validations")
    print("------------------------------")
    print(total_modules)
    # for c in missing_chapters:
    #     print(f"- {c}")

    print()


if __name__ == "__main__":
    main()