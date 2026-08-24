import pandas as pd
from pathlib import Path
import re


# ============================================================
# LOAD DATA
# ============================================================

INPUT_PATH = Path(
    "data/processed/meddialog_analyzed.csv"
)

OUTPUT_PATH = Path(
    "data/processed/meddialog_scored.csv"
)

df = pd.read_csv(INPUT_PATH)


# ============================================================
# TEXT
# ============================================================

df["patient_text"] = (
    df["patient_text"]
    .fillna("")
    .astype(str)
)

df["description"] = (
    df["description"]
    .fillna("")
    .astype(str)
)

df["patient_full_text"] = (
    df["description"] + " " +
    df["patient_text"]
).str.lower()


# ============================================================
# INFORMATION CATEGORIES
# ============================================================

context_patterns = {

    "family": [
        r"\bmy (wife|husband|son|daughter|child|children|baby|mother|father|parent|parents)\b",
        r"\bmy family\b",
        r"\bfamily\b"
    ],

    "occupation": [
        r"\bwork\b",
        r"\bworking\b",
        r"\bjob\b",
        r"\boffice\b",
        r"\bemploy(ed|ee)\b",
        r"\bnurse\b",
        r"\bdoctor\b",
        r"\bpharmacist\b",
        r"\bteacher\b"
    ],

    "medical_history": [
        r"\bhistory of\b",
        r"\bpreviously\b",
        r"\bprevious\b",
        r"\bdiagnosed with\b",
        r"\bdiabetes\b",
        r"\basthma\b",
        r"\bcopd\b",
        r"\bhypertension\b",
        r"\bcancer\b",
        r"\bimmune\b",
        r"\bimmunocompromised\b"
    ],

    "medication": [
        r"\bmedication\b",
        r"\bmedicine\b",
        r"\bantibiotic\b",
        r"\bantibiotics\b",
        r"\btablet\b",
        r"\btablets\b",
        r"\bdrug\b",
        r"\btaking\b",
        r"\bdexamethasone\b"
    ],

    "timeline": [
        r"\byesterday\b",
        r"\btoday\b",
        r"\blast week\b",
        r"\blast month\b",
        r"\bfor \d+ days\b",
        r"\bfor \d+ weeks\b",
        r"\bfor \d+ months\b",
        r"\bsince\b",
        r"\bago\b"
    ],

    "exposure": [
        r"\bcontact\b",
        r"\bexposed\b",
        r"\bexposure\b",
        r"\btravel(l)?ed\b",
        r"\btravel\b",
        r"\bworkplace\b",
        r"\bhospital\b"
    ],

    "pregnancy": [
        r"\bpregnan(t|cy)\b",
        r"\bmiscarriage\b",
        r"\bmiscarried\b",
        r"\bpcos\b",
        r"\btrimester\b",
        r"\bbreastfeed(ing)?\b"
    ]
}


emotional_patterns = {

    "fear": [
        r"\bscared\b",
        r"\bafraid\b",
        r"\bfrightened\b",
        r"\bterrified\b",
        r"\bpetrified\b",
        r"\bfear\b"
    ],

    "worry": [
        r"\bworried\b",
        r"\bworry\b",
        r"\bworrying\b",
        r"\bconcerned\b",
        r"\bconcern\b"
    ],

    "anxiety": [
        r"\banxious\b",
        r"\banxiety\b",
        r"\bpanic\b",
        r"\bpanicked\b",
        r"\bnervous\b"
    ],

    "distress": [
        r"\boverwhelmed\b",
        r"\bupset\b",
        r"\bhelpless\b",
        r"\bdesperate\b",
        r"\bsad\b",
        r"\bdepressed\b"
    ]
}


clinical_patterns = {

    "symptoms": [
        r"\bcough\b",
        r"\bfever\b",
        r"\bpain\b",
        r"\bheadache\b",
        r"\bnausea\b",
        r"\bvomit\b",
        r"\bdizziness\b",
        r"\bbreathing\b",
        r"\bshortness of breath\b"
    ],

    "severity": [
        r"\bsevere\b",
        r"\bmild\b",
        r"\bmoderate\b",
        r"\bworsening\b",
        r"\bdifficulty\b"
    ],

    "diagnosis": [
        r"\bdiagnosed\b",
        r"\bdiagnosis\b",
        r"\bpneumonia\b",
        r"\bcovid\b",
        r"\bcoronavirus\b",
        r"\binfluenza\b",
        r"\basthma\b"
    ]
}


# ============================================================
# COUNT PATTERNS
# ============================================================

def count_matches(text, patterns):

    count = 0

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        count += len(matches)

    return count


def category_score(text, category_dict):

    total = 0

    detected = []

    for category, patterns in category_dict.items():

        score = count_matches(
            text,
            patterns
        )

        if score > 0:

            detected.append(category)

            total += score

    return total, detected


# ============================================================
# CALCULATE SCORES
# ============================================================

results = []

for text in df["patient_full_text"]:

    context_score, context_types = category_score(
        text,
        context_patterns
    )

    emotion_score, emotion_types = category_score(
        text,
        emotional_patterns
    )

    clinical_score, clinical_types = category_score(
        text,
        clinical_patterns
    )

    results.append({
        "context_score": context_score,
        "emotion_score": emotion_score,
        "clinical_score": clinical_score,
        "context_types": ", ".join(context_types),
        "emotion_types": ", ".join(emotion_types),
        "clinical_types": ", ".join(clinical_types)
    })


scores = pd.DataFrame(results)


# ============================================================
# MERGE
# ============================================================

df = pd.concat(
    [
        df.reset_index(drop=True),
        scores
    ],
    axis=1
)


# ============================================================
# OVERALL INFORMATION SCORE
# ============================================================

df["refined_information_score"] = (
    df["context_score"]
    + df["emotion_score"]
)


# ============================================================
# EMOTIONAL FLAG
# ============================================================

df["refined_emotional"] = (
    df["emotion_score"] > 0
)


# ============================================================
# CONTEXTUAL FLAG
# ============================================================

df["refined_contextual"] = (
    df["context_score"] > 0
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    "refined_information_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("REFINED INFORMATION ANALYSIS")
print("=" * 60)

print(
    "Total conversations:",
    len(df)
)

print(
    "\nEmotionally detected:",
    df["refined_emotional"].sum()
)

print(
    "Contextually detected:",
    df["refined_contextual"].sum()
)


print("\n" + "=" * 60)
print("SCORE DISTRIBUTION")
print("=" * 60)

print(
    df["refined_information_score"].describe()
)


print("\n" + "=" * 60)
print("TOP 20 CASES")
print("=" * 60)

for i, row in df.head(20).iterrows():

    print("\nID:", i)

    print(
        "Score:",
        row["refined_information_score"]
    )

    print(
        "Context:",
        row["context_types"]
    )

    print(
        "Emotion:",
        row["emotion_types"]
    )

    print(
        "Description:",
        row["description"]
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(OUTPUT_PATH)