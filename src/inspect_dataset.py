import json
import os

BASE_PATH = "data/raw/meddialog"

files = ["train.json", "validation.json", "test.json"]

for filename in files:
    path = os.path.join(BASE_PATH, filename)

    print("\n" + "=" * 60)
    print(filename.upper())
    print("=" * 60)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Number of conversations:", len(data))

    print("\nFirst conversation:")
    print(data[0])