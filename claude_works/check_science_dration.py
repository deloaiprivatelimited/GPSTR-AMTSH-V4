import json
from pathlib import Path

AUDIO_FOLDER = Path("claude_works/audio_science")

def get_total_science_duration():
    total_duration = 0.0

    for chapter in AUDIO_FOLDER.iterdir():
        if not chapter.is_dir():
            continue

        for concept in chapter.iterdir():   # <-- concept level
            if not concept.is_dir():
                continue

            timeline_path = concept / "timeline.json"

            if timeline_path.exists():
                try:
                    data = json.loads(timeline_path.read_text(encoding="utf-8"))
                    total_duration += data.get("total_duration", 0)
                except Exception as e:
                    print(f"Error reading {timeline_path}: {e}")

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"✅ Total Science Duration: {hours}h {minutes}m {seconds}s")
    return total_duration


if __name__ == "__main__":
    get_total_science_duration()