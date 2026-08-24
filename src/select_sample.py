import pandas as pd
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

INPUT_PATH = Path(
    "data/processed/meddialog_analyzed.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("SAMPLE SELECTION")
print("=" * 60)

print("Total available conversations:", len(df))


# ============================================================
# CREATE CONTEXT / EMOTION SCORE
# ============================================================

df["context_emotion_score"] = (
    df["context_keyword_count"]
    + df["emotional_keyword_count"]
)


# ============================================================
# RANK CASES
# ============================================================

# Sort from highest contextual/emotional information
# to lowest.

df_sorted = df.sort_values(
    by=[
        "context_emotion_score",
        "total_word_count"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# ============================================================
# SELECT RICH GROUP
# ============================================================

rich_group = df_sorted.head(150).copy()

rich_group["sample_group"] = "contextually_rich"


# ============================================================
# SELECT SPARSE GROUP
# ============================================================

sparse_group = df_sorted.tail(150).copy()

sparse_group["sample_group"] = "contextually_sparse"


# ============================================================
# COMBINE
# ============================================================

sample = pd.concat(
    [
        rich_group,
        sparse_group
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

sample = sample.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# ADD EXPERIMENT ID
# ============================================================

sample.insert(
    0,
    "case_id",
    range(1, len(sample) + 1)
)


# ============================================================
# VERIFY SAMPLE SIZE
# ============================================================

print("\nSelected conversations:", len(sample))

print("\nSample distribution:")
print(
    sample["sample_group"].value_counts()
)


# ============================================================
# SHOW SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("SCORE DISTRIBUTION")
print("=" * 60)

print(
    sample.groupby("sample_group")[
        "context_emotion_score"
    ].describe()
)


# ============================================================
# SHOW CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CATEGORY DISTRIBUTION")
print("=" * 60)

print(
    pd.crosstab(
        sample["sample_group"],
        sample["primary_category"]
    )
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n" + "=" * 60)
print("RICH GROUP EXAMPLES")
print("=" * 60)

for _, row in rich_group.head(10).iterrows():

    print("\nCase:", row.name)

    print(
        "Score:",
        row["context_emotion_score"]
    )

    print(
        "Description:",
        row["description"]
    )


print("\n" + "=" * 60)
print("SPARSE GROUP EXAMPLES")
print("=" * 60)

for _, row in sparse_group.head(10).iterrows():

    print("\nCase:", row.name)

    print(
        "Score:",
        row["context_emotion_score"]
    )

    print(
        "Description:",
        row["description"]
    )


# ============================================================
# SAVE SAMPLE
# ============================================================

OUTPUT_PATH = (
    OUTPUT_DIR /
    "sample_300_conversations.csv"
)

sample.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 60)
print("SAMPLE SAVED")
print("=" * 60)

print(OUTPUT_PATH)