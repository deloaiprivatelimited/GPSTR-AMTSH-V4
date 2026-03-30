import os
import json

VALIDATION_DIR = "validation_reports"
MASTER_DATA_DIR = "master_data"

def process_files():
    for file in os.listdir(VALIDATION_DIR):
        if not file.endswith("_validation.json"):
            continue

        validation_path = os.path.join(VALIDATION_DIR, file)

        try:
            with open(validation_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            recommendation = data.get("release_recommendation", "").strip()

            if recommendation != "READY_TO_GO":
                base_name = file.replace("_validation.json", "")

                master_file = os.path.join(MASTER_DATA_DIR, base_name + ".txt")

                if os.path.exists(validation_path):
                    os.remove(validation_path)
                    print(f"Deleted validation: {validation_path}")

                if os.path.exists(master_file):
                    os.remove(master_file)
                    print(f"Deleted master data: {master_file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    process_files()