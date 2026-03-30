import json

with open("data.json") as f:
    data = json.load(f)

total = sum(len(chapters) for chapters in data.values())

print("Total chapters:", total)