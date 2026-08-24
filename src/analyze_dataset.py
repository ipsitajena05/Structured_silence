import pandas as pd
from pathlib import Path
import re

# ============================================================
# LOAD DATA
# ============================================================

INPUT_PATH = Path("data/processed/meddialog_clean.csv")

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print("Total conversations:", len(df))


# ============================================================
# COMBINE TEXT
# ============================================================

df["full_text"] = (
    df["description"].fillna("").astype(str)
    + " "
    + df["patient_text"].fillna("").astype(str)
)


# ============================================================
# CATEGORY KEYWORDS
# ============================================================

categories = {

    "respiratory": [
        "cough",
        "breath",
        "breathing",
        "pneumonia",
        "asthma",
        "flu",
        "cold",
        "throat",
        "lung",
        "covid",
        "coronavirus",
        "congestion",
        "wheezing",
        "respiratory"
    ],

    "cardiovascular": [
        "chest pain",
        "heart",
        "blood pressure",
        "hypertension",
        "palpitation",
        "palpitations",
        "cardiac",
        "heartbeat",
        "heart rate"
    ],

    "reproductive_health": [
        "pregnant",
        "pregnancy",
        "miscarriage",
        "period",
        "menstrual",
        "ovulation",
        "fertility",
        "fertility",
        "pcos",
        "contraception",
        "breastfeeding",
        "cervix"
    ],

    "mental_health": [
        "anxiety",
        "anxious",
        "depression",
        "depressed",
        "panic",
        "stress",
        "suicidal",
        "suicide",
        "mental health",
        "insomnia",
        "sleep"
    ],

    "chronic_pain": [
        "chronic pain",
        "back pain",
        "neck pain",
        "joint pain",
        "arthritis",
        "migraine",
        "headache",
        "fibromyalgia"
    ]
}


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_categories(text):

    text = text.lower()

    detected = []

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword in text:
                detected.append(category)
                break

    return detected


df["categories"] = df["full_text"].apply(
    detect_categories
)


# ============================================================
# PRIMARY CATEGORY
# ============================================================

def primary_category(category_list):

    if not category_list:
        return "other"

    return category_list[0]


df["primary_category"] = df["categories"].apply(
    primary_category
)


# ============================================================
# CATEGORY COUNTS
# ============================================================

print("\n" + "=" * 60)
print("PRIMARY CATEGORY DISTRIBUTION")
print("=" * 60)

print(
    df["primary_category"].value_counts()
)


# ============================================================
# EMOTIONAL / CONTEXTUAL LANGUAGE
# ============================================================

emotional_keywords = [
    "worried",
    "worry",
    "scared",
    "afraid",
    "fear",
    "anxious",
    "anxiety",
    "concerned",
    "concern",
    "stress",
    "stressed",
    "panic",
    "upset",
    "terrified",
    "frightened",
    "nervous",
    "sad",
    "depressed",
    "overwhelmed",
    "confused",
    "helpless",
    "desperate",
    "please help",
    "i'm scared",
    "i am scared"
]


def count_emotional_terms(text):

    text = text.lower()

    count = 0

    for keyword in emotional_keywords:

        count += text.count(keyword)

    return count


df["emotional_keyword_count"] = df["full_text"].apply(
    count_emotional_terms
)


df["emotionally_loaded"] = (
    df["emotional_keyword_count"] > 0
)


# ============================================================
# CONTEXTUAL INFORMATION
# ============================================================

context_keywords = [
    "my wife",
    "my husband",
    "my daughter",
    "my son",
    "my child",
    "my baby",
    "my mother",
    "my father",
    "my family",
    "work",
    "job",
    "school",
    "pregnant",
    "children",
    "child",
    "hospital",
    "doctor",
    "previously",
    "history",
    "diagnosed",
    "medication",
    "treatment",
    "years ago",
    "last week",
    "yesterday",
    "months ago"
]


def count_context_terms(text):

    text = text.lower()

    count = 0

    for keyword in context_keywords:

        count += text.count(keyword)

    return count


df["context_keyword_count"] = df["full_text"].apply(
    count_context_terms
)


# ============================================================
# TEXT LENGTH / COMPLEXITY
# ============================================================

df["patient_word_count"] = pd.to_numeric(
    df["patient_word_count"],
    errors="coerce"
).fillna(0)


df["doctor_word_count"] = pd.to_numeric(
    df["doctor_word_count"],
    errors="coerce"
).fillna(0)


df["total_word_count"] = (
    df["patient_word_count"]
    + df["doctor_word_count"]
)


# ============================================================
# INFORMATION-RICHNESS SCORE
# ============================================================

df["information_richness_score"] = (
    df["context_keyword_count"]
    + df["emotional_keyword_count"]
)


# ============================================================
# PRINT EMOTIONAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("EMOTIONAL / CONTEXTUAL ANALYSIS")
print("=" * 60)

print(
    "Emotionally loaded conversations:",
    df["emotionally_loaded"].sum()
)

print(
    "Conversations with contextual information:",
    (df["context_keyword_count"] > 0).sum()
)


# ============================================================
# MOST INFORMATION-RICH CASES
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 INFORMATION-RICH CONVERSATIONS")
print("=" * 60)

top_cases = df.sort_values(
    "information_richness_score",
    ascending=False
).head(20)


for index, row in top_cases.iterrows():

    print("\nID:", index)

    print(
        "Category:",
        row["primary_category"]
    )

    print(
        "Information score:",
        row["information_richness_score"]
    )

    print(
        "Description:",
        row["description"]
    )


# ============================================================
# SAVE ANALYZED DATA
# ============================================================

OUTPUT_PATH = Path(
    "data/processed/meddialog_analyzed.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 60)
print("ANALYSIS SAVED")
print("=" * 60)

print(OUTPUT_PATH)