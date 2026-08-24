import json
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/raw/meddialog")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json(filename, split_name):

    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:

        rows.append({
            "split": split_name,
            "description": item.get("description", ""),
            "utterances": item.get("utterances", [])
        })

    return rows


# ============================================================
# LOAD ALL SPLITS
# ============================================================

train = load_json("train.json", "train")
validation = load_json("validation.json", "validation")
test = load_json("test.json", "test")

all_data = train + validation + test

df = pd.DataFrame(all_data)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)

print("Total conversations:", len(df))

print("\nSplit distribution:")
print(df["split"].value_counts())


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

# Convert utterances lists to strings temporarily
# so pandas can check for duplicate conversations.
df["_utterances_for_duplicate_check"] = df["utterances"].apply(
    lambda x: json.dumps(x, ensure_ascii=False)
)

df = df.drop_duplicates(
    subset=["description", "_utterances_for_duplicate_check"]
).reset_index(drop=True)

# Remove temporary column
df = df.drop(
    columns=["_utterances_for_duplicate_check"]
)

after = len(df)

print("\nDuplicate conversations removed:", before - after)
print("Remaining conversations:", after)


# ============================================================
# PARSE PATIENT / DOCTOR UTTERANCES
# ============================================================

def extract_speakers(utterances):

    patient_messages = []
    doctor_messages = []

    for utterance in utterances:

        utterance = str(utterance)

        lower = utterance.lower()

        if lower.startswith("patient:"):
            patient_messages.append(
                utterance[len("patient:"):].strip()
            )

        elif lower.startswith("doctor:"):
            doctor_messages.append(
                utterance[len("doctor:"):].strip()
            )

    return patient_messages, doctor_messages


df["patient_messages"] = df["utterances"].apply(
    lambda x: extract_speakers(x)[0]
)

df["doctor_messages"] = df["utterances"].apply(
    lambda x: extract_speakers(x)[1]
)


# ============================================================
# CONVERSATION STATISTICS
# ============================================================

df["num_utterances"] = df["utterances"].apply(len)

df["num_patient_turns"] = df["patient_messages"].apply(len)

df["num_doctor_turns"] = df["doctor_messages"].apply(len)

df["patient_text"] = df["patient_messages"].apply(
    lambda x: " ".join(x)
)

df["doctor_text"] = df["doctor_messages"].apply(
    lambda x: " ".join(x)
)


# ============================================================
# TEXT LENGTH
# ============================================================

df["patient_word_count"] = df["patient_text"].apply(
    lambda x: len(x.split())
)

df["doctor_word_count"] = df["doctor_text"].apply(
    lambda x: len(x.split())
)

df["total_word_count"] = (
    df["patient_word_count"] +
    df["doctor_word_count"]
)


# ============================================================
# QUALITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("QUALITY CHECK")
print("=" * 60)

print(
    "Conversations with no patient message:",
    (df["num_patient_turns"] == 0).sum()
)

print(
    "Conversations with no doctor message:",
    (df["num_doctor_turns"] == 0).sum()
)

print(
    "Average utterances:",
    round(df["num_utterances"].mean(), 2)
)

print(
    "Average patient words:",
    round(df["patient_word_count"].mean(), 2)
)

print(
    "Average doctor words:",
    round(df["doctor_word_count"].mean(), 2)
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

output_path = OUTPUT_DIR / "meddialog_clean.csv"

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(output_path)