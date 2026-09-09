import pandas as pd
import ollama
from transformers import pipeline
from pathlib import Path

# ============================================================
# 1. PATHS & SETUP
# ============================================================
INPUT_FILE = Path("data/processed/final_300_conversations.csv")
OUTPUT_FILE = Path("data/results/model_baseline_comparisons.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} conversations from {INPUT_FILE}")

# Initialize BioGPT pipeline
print("Initializing BioGPT pipeline from Hugging Face...")
biogpt_pipe = pipeline("text-generation", model="microsoft/biogpt", max_new_tokens=100)

# Define standard clinical summarization prompt
SYSTEM_PROMPT = """You are a clinical AI assistant. Summarize the following patient query concise and accurately, highlighting primary clinical facts, symptoms, and recommendations."""

# ============================================================
# 2. EXECUTE GENERATIONS & CAPTURE OUTPUTS
# ============================================================
results = []

print("\nStarting Stage 02 Model Audit...")
for idx, row in df.iterrows():
    # Extract patient query
    patient_input = str(row.get('utterances', row.get('description', ''))).strip()
    
    if not patient_input:
        continue

    # --- LLaMA-3.2 (Ollama) ---
    try:
        llama_res = ollama.chat(
            model='llama3.2:latest',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': patient_input}
            ]
        )
        llama_output = llama_res['message']['content']
    except Exception as e:
        llama_output = f"Error generating LLaMA response: {e}"

    # --- BioGPT (Hugging Face) ---
    try:
        biogpt_res = biogpt_pipe(patient_input[:512]) # Truncate to fit sequence length
        biogpt_output = biogpt_res[0]['generated_text']
    except Exception as e:
        biogpt_output = f"Error generating BioGPT response: {e}"

    # Append row record
    results.append({
        'id': row.get('id', idx),
        'patient_input': patient_input,
        'llama3_2_output': llama_output,
        'biogpt_output': biogpt_output
    })

    # Progress tracking
    if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
        print(f"Processed {idx + 1}/{len(df)} conversations...")

# ============================================================
# 3. SAVE RESULTS
# ============================================================
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("STAGE 02 COMPLETE")
print("=" * 60)
print(f"Outputs successfully saved to: {OUTPUT_FILE}")