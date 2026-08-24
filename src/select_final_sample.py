import pandas as pd
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

INPUT_PATH = Path(
    "data/processed/meddialog_scored.csv"
)

OUTPUT_PATH = Path(
    "data/processed/final_300_conversations.csv"
)

RANDOM_SEED = 42


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("FINAL EXPERIMENTAL SAMPLE")
print("=" * 60)

print("Available conversations:", len(df))


# ============================================================
# SORT BY REFINED INFORMATION SCORE
# ============================================================

df = df.sort_values(
    by=[
        "refined_information_score",
        "total_word_count"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# ============================================================
# SELECT 150 HIGH-INFORMATION CASES
# ============================================================

rich = df.head(150).copy()

rich["sample_group"] = "information_rich"


# ============================================================
# SELECT 150 LOW-INFORMATION CASES
# ============================================================

sparse = df.tail(150).copy()

sparse["sample_group"] = "information_sparse"


# ============================================================
# COMBINE
# ============================================================

sample = pd.concat(
    [rich, sparse],
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
# CASE ID
# ============================================================

sample.insert(
    0,
    "case_id",
    range(1, len(sample) + 1)
)


# ============================================================
# VERIFY
# ============================================================

print("\nFinal sample size:", len(sample))

print("\nGroup distribution:")
print(
    sample["sample_group"].value_counts()
)


print("\n" + "=" * 60)
print("INFORMATION SCORE")
print("=" * 60)

print(
    sample.groupby("sample_group")[
        "refined_information_score"
    ].describe()
)


print("\n" + "=" * 60)
print("EMOTIONAL DISTRIBUTION")
print("=" * 60)

print(
    sample.groupby("sample_group")[
        "refined_emotional"
    ].sum()
)


print("\n" + "=" * 60)
print("CONTEXTUAL DISTRIBUTION")
print("=" * 60)

print(
    sample.groupby("sample_group")[
        "refined_contextual"
    ].sum()
)


# ============================================================
# CATEGORY DISTRIBUTION
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
# SAVE
# ============================================================

sample.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 60)
print("FINAL SAMPLE SAVED")
print("=" * 60)

print(OUTPUT_PATH)