import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/processed/meddialog_scored.csv")
OUTPUT_DIR = Path("data/visualizations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Total conversations: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 1. CATEGORY DISTRIBUTION
# ============================================================

if "primary_category" in df.columns:

    category_counts = df["primary_category"].value_counts()

    plt.figure(figsize=(10, 6))

    category_counts.plot(kind="bar")

    plt.title("Distribution of Medical Conversation Categories")
    plt.xlabel("Medical Category")
    plt.ylabel("Number of Conversations")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "category_distribution.png",
        dpi=300
    )

    plt.close()

    print("Saved: category_distribution.png")


# ============================================================
# 2. INFORMATION SCORE DISTRIBUTION
# ============================================================

score_column = None

if "refined_information_score" in df.columns:
    score_column = "refined_information_score"

elif "information_richness_score" in df.columns:
    score_column = "information_richness_score"

elif "information_score" in df.columns:
    score_column = "information_score"


if score_column:

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[score_column].dropna(),
        bins=15
    )

    plt.title("Distribution of Information Scores")
    plt.xlabel("Information Score")
    plt.ylabel("Number of Conversations")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "information_score_distribution.png",
        dpi=300
    )

    plt.close()

    print("Saved: information_score_distribution.png")


# ============================================================
# 3. CONTEXTUALLY RICH VS SPARSE
# ============================================================

# The final dataset may not contain sample_group.
# Therefore, create the groups using refined_information_score.

if "refined_information_score" in df.columns:

    rich_count = (
        df["refined_information_score"] > 0
    ).sum()

    sparse_count = (
        df["refined_information_score"] == 0
    ).sum()

    group_counts = pd.Series({
        "Contextually rich": rich_count,
        "Contextually sparse": sparse_count
    })

    plt.figure(figsize=(8, 6))

    group_counts.plot(kind="bar")

    plt.title("Contextually Rich vs Sparse Conversations")
    plt.xlabel("Conversation Group")
    plt.ylabel("Number of Conversations")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "context_rich_vs_sparse.png",
        dpi=300
    )

    plt.close()

    print("Saved: context_rich_vs_sparse.png")


# ============================================================
# 4. EMOTION PRESENCE
# ============================================================

emotion_present = None


# ------------------------------------------------------------
# Preferred: refined_emotional
# ------------------------------------------------------------

if "refined_emotional" in df.columns:

    values = df["refined_emotional"]

    # Handle boolean values
    if values.dtype == bool:

        emotion_present = values

    else:

        values = values.fillna("").astype(str).str.strip().str.lower()

        emotion_present = values.isin([
            "true",
            "1",
            "yes",
            "y",
            "detected"
        ])


# ------------------------------------------------------------
# If refined_emotional is unavailable, use emotion_types
# ------------------------------------------------------------

elif "emotion_types" in df.columns:

    values = (
        df["emotion_types"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    emotion_present = ~values.isin([
        "",
        "[]",
        "none",
        "nan",
        "null"
    ])


# ------------------------------------------------------------
# Final fallback
# ------------------------------------------------------------

elif "emotionally_loaded" in df.columns:

    values = df["emotionally_loaded"]

    if values.dtype == bool:

        emotion_present = values

    else:

        values = (
            values
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        emotion_present = values.isin([
            "true",
            "1",
            "yes",
            "y",
            "detected"
        ])


# ------------------------------------------------------------
# Create graph
# ------------------------------------------------------------

if emotion_present is not None:

    emotion_counts = pd.Series({
        "Emotion detected": int(emotion_present.sum()),
        "No emotion detected": int((~emotion_present).sum())
    })

    plt.figure(figsize=(8, 6))

    emotion_counts.plot(kind="bar")

    plt.title("Presence of Emotional Information")
    plt.xlabel("")
    plt.ylabel("Number of Conversations")

    plt.xticks(rotation=0)

    # Add values above bars
    for i, value in enumerate(emotion_counts):

        plt.text(
            i,
            value + 5,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "emotion_presence.png",
        dpi=300
    )

    plt.close()

    print("Saved: emotion_presence.png")

    print()
    print("EMOTION SUMMARY")
    print(f"Emotion detected: {emotion_counts['Emotion detected']}")
    print(f"No emotion detected: {emotion_counts['No emotion detected']}")


# ============================================================
# 5. CONTEXT TYPE FREQUENCY
# ============================================================

context_column = None

for column in [
    "context_types",
    "context",
    "contexts"
]:

    if column in df.columns:
        context_column = column
        break


if context_column:

    context_counts = {}

    for value in df[context_column].fillna(""):

        # Contexts may be stored like:
        # occupation, medication, timeline
        # or ['occupation', 'medication']

        value = str(value).strip()

        # Remove list brackets
        value = (
            value
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )

        items = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

        for item in items:

            context_counts[item] = (
                context_counts.get(item, 0) + 1
            )

    if context_counts:

        context_series = (
            pd.Series(context_counts)
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))

        context_series.plot(kind="bar")

        plt.title("Frequency of Contextual Information")
        plt.xlabel("Context Type")
        plt.ylabel("Number of Conversations")

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "context_type_frequency.png",
            dpi=300
        )

        plt.close()

        print("Saved: context_type_frequency.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print("Charts saved to:")
print(OUTPUT_DIR)