import pandas as pd
import ollama
from transformers import pipeline
from pathlib import Path

# 1. Paths Setup
INPUT_FILE = Path("data/processed/final_300_conversations.csv")
OUTPUT_FILE = Path("data/results/model_baseline_comparisons.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 2. Load Dataset
df = pd.read_csv(INPUT_FILE)

# 3. Load BioGPT from Hugging Face
print("Loading BioGPT pipeline...")
biogpt_generator = pipeline("text-generation", model="microsoft/biogpt", max_new_tokens=100)

SYSTEM_PROMPT = """Summarize the medical conversation, highlighting clinical facts and symptoms."""

results = []

# 4. Loop through sample conversations (testing first 10 for quick verification)
for idx, row in df.head(10).iterrows():
    patient_text = str(row.get('utterances', row.get('description', '')))
    if not patient_text.strip():
        continue

    # --- Run LLaMA-3.2 (Ollama) ---
    llama_res = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': patient_text}
        ]
    )
    llama_out = llama_res['message']['content']

    # --- Run BioGPT (Hugging Face) ---
    try:
        biogpt_res = biogpt_generator(patient_text[:512]) # Truncate for standard context window
        biogpt_out = biogpt_res[0]['generated_text']
    except Exception as e:
        biogpt_out = f"Error: {e}"

    results.append({
        'id': row.get('id', idx),
        'patient_input': patient_text,
        'llama3_2_output': llama_out,
        'biogpt_output': biogpt_out
    })
    print(f"Completed conversation {idx + 1}")

# 5. Save Model Output Comparison
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"Comparison complete! Saved to {OUTPUT_FILE}")