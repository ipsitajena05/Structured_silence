import pandas as pd
import ast
import re
from pathlib import Path


INPUT_FILE = Path("data/processed/final_300_conversations.csv")
OUTPUT_FILE = Path("data/processed/structured_300.csv")


# ============================================================
# KEYWORD DICTIONARIES
# ============================================================

CONTEXT_KEYWORDS = {

    "family": [
        "wife", "husband", "mother", "father", "son", "daughter",
        "child", "children", "baby", "family", "parent", "sister",
        "brother", "partner"
    ],

    "occupation": [
        "work", "worker", "job", "office", "nurse", "doctor",
        "pharmacist", "teacher", "employee", "hospital", "workplace"
    ],

    "medical_history": [
        "diabetes", "asthma", "cancer", "copd", "hypertension",
        "pneumonia", "pregnant", "pregnancy", "pcos", "heart",
        "chronic", "disease", "immunocompromised", "immune"
    ],

    "medication": [
        "medicine", "medication", "antibiotic", "antibiotics",
        "tablet", "pill", "drug", "levofloxacin", "azithromycin",
        "amoxicillin", "dexamethasone", "ibuprofen"
    ],

    "timeline": [
        "today", "yesterday", "tomorrow", "week", "weeks",
        "month", "months", "day", "days", "ago", "since",
        "for the past", "recently"
    ],

    "exposure": [
        "contact", "exposed", "exposure", "travel", "travelled",
        "travelling", "returned", "positive", "infected",
        "patient", "covid", "coronavirus"
    ],

    "pregnancy": [
        "pregnant", "pregnancy", "trimester", "maternity",
        "miscarriage", "miscarried", "fertility"
    ]
}


EMOTION_KEYWORDS = {

    "fear": [
        "scared", "afraid", "terrified", "frightened",
        "fear", "petrified"
    ],

    "worry": [
        "worried", "worry", "concerned", "concern",
        "anxious", "anxiety"
    ],

    "distress": [
        "panic", "panicking", "upset", "desperate",
        "distressed", "wreck"
    ],

    "uncertainty": [
        "not sure", "unsure", "could it", "should i",
        "what if", "wondering", "possibly"
    ]
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_patient_text(row):
    """
    Extract only the patient utterances.
    """

    utterances = row.get("utterances", [])

    if isinstance(utterances, str):
        try:
            utterances = ast.literal_eval(utterances)
        except Exception:
            utterances = [utterances]

    patient_messages = []

    for utterance in utterances:

        utterance = str(utterance).strip()

        if utterance.lower().startswith("patient:"):
            patient_messages.append(
                utterance[len("patient:"):].strip()
            )

    return " ".join(patient_messages)


def detect_categories(text, dictionary):

    detected = []

    for category, keywords in dictionary.items():

        for keyword in keywords:

            if keyword in text:
                detected.append(category)
                break

    return sorted(set(detected))


def count_words(text):
    return len(text.split())


def build_context_string(contexts):

    if not contexts:
        return "none"

    return ", ".join(contexts)


def build_emotion_string(emotions):

    if not emotions:
        return "none"

    return ", ".join(emotions)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STRUCTURED SILENCE - FEATURE EXTRACTION")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded conversations: {len(df)}")

    # --------------------------------------------------------
    # Patient text
    # --------------------------------------------------------

    df["patient_text"] = df.apply(
        extract_patient_text,
        axis=1
    )

    df["patient_text"] = df["patient_text"].apply(clean_text)

    # --------------------------------------------------------
    # Context extraction
    # --------------------------------------------------------

    df["context_features"] = df["patient_text"].apply(
        lambda x: detect_categories(
            x,
            CONTEXT_KEYWORDS
        )
    )

    # --------------------------------------------------------
    # Emotion extraction
    # --------------------------------------------------------

    df["emotion_features"] = df["patient_text"].apply(
        lambda x: detect_categories(
            x,
            EMOTION_KEYWORDS
        )
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    df["context_count"] = df["context_features"].apply(len)

    df["emotion_count"] = df["emotion_features"].apply(len)

    df["patient_word_count"] = df["patient_text"].apply(
        count_words
    )

    # --------------------------------------------------------
    # Structured representations
    # --------------------------------------------------------

    df["context_string"] = df["context_features"].apply(
        build_context_string
    )

    df["emotion_string"] = df["emotion_features"].apply(
        build_emotion_string
    )

    # --------------------------------------------------------
    # Information score
    # --------------------------------------------------------

    df["information_score"] = (
        df["context_count"] * 2
        + df["emotion_count"]
    )

    # --------------------------------------------------------
    # Contextually rich / sparse
    # --------------------------------------------------------

    df["context_group"] = df["context_count"].apply(
        lambda x:
            "contextually_rich"
            if x > 0
            else "contextually_sparse"
    )

    # --------------------------------------------------------
    # Emotionally loaded
    # --------------------------------------------------------

    df["emotionally_loaded"] = (
        df["emotion_count"] > 0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        "\nContext distribution:"
    )

    print(
        df["context_group"].value_counts()
    )

    print(
        "\nEmotionally loaded:"
    )

    print(
        df["emotionally_loaded"].sum()
    )

    print(
        "\nAverage information score:"
    )

    print(
        round(
            df["information_score"].mean(),
            2
        )
    )

    print(
        "\nSaved:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()