import pandas as pd
import re
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/structured_300.csv"
)

OUTPUT_FILE = Path(
    "data/processed/context_comparison.csv"
)


CONTEXT_PATTERNS = [
    r"\bmy wife\b",
    r"\bmy husband\b",
    r"\bmy child\b",
    r"\bmy children\b",
    r"\bmy family\b",
    r"\bmy mother\b",
    r"\bmy father\b",
    r"\bi work\b",
    r"\bat work\b",
    r"\bat my workplace\b",
    r"\bi am a nurse\b",
    r"\bi am a doctor\b",
    r"\bi am pregnant\b",
    r"\bi have diabetes\b",
    r"\bi have asthma\b",
    r"\bi have copd\b",
    r"\bi have cancer\b",
    r"\bi have hypertension\b",
    r"\bi take\b",
    r"\bi am taking\b",
    r"\bi was diagnosed\b",
    r"\bi was exposed\b",
    r"\bi had contact\b",
    r"\bi travelled\b",
    r"\bi traveled\b"
]


EMOTION_PATTERNS = [
    r"\bscared\b",
    r"\bterrified\b",
    r"\bafraid\b",
    r"\bworried\b",
    r"\bworry\b",
    r"\banxious\b",
    r"\banxiety\b",
    r"\bpetrified\b",
    r"\bpanic\b",
    r"\bfrightened\b",
    r"\bconcerned\b"
]


def find_matches(text, patterns):

    found = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text.lower()
        )

        if matches:
            found.extend(matches)

    return sorted(set(found))


def main():

    df = pd.read_csv(INPUT_FILE)

    results = []

    for index, row in df.iterrows():

        text = str(
            row["patient_text"]
        ).lower()

        contexts = find_matches(
            text,
            CONTEXT_PATTERNS
        )

        emotions = find_matches(
            text,
            EMOTION_PATTERNS
        )

        context_count = len(contexts)
        emotion_count = len(emotions)

        results.append({

            "id": index,

            "patient_text": text,

            "context_features":
                row["context_string"],

            "emotion_features":
                row["emotion_string"],

            "detected_context_phrases":
                ", ".join(contexts),

            "detected_emotion_phrases":
                ", ".join(emotions),

            "context_count":
                context_count,

            "emotion_count":
                emotion_count,

            "context_information_score":
                context_count + emotion_count,

            "contextually_rich":
                context_count > 0,

            "emotionally_loaded":
                emotion_count > 0
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("CONTEXT COMPARISON")
    print("=" * 60)

    print(
        "\nTotal conversations:",
        len(result_df)
    )

    print(
        "\nContextually rich:",
        result_df["contextually_rich"].sum()
    )

    print(
        "Emotionally loaded:",
        result_df["emotionally_loaded"].sum()
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()