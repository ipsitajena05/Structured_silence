import pandas as pd
import ollama
from pathlib import Path

# 1. Setup paths
INPUT_FILE = Path("data/processed/final_300_conversations.csv")
OUTPUT_FILE = Path("data/results/llama_baseline_outputs.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} conversations for baseline evaluation.")

# 3. Standard Clinical AI System Prompt
SYSTEM_PROMPT = """You are an automated medical dialogue assistant. 
Summarize or respond to the patient's query concisely, highlighting the primary clinical facts, symptoms, and recommendations."""

results = []

# 4. Run LLaMA on patient utterances
for idx, row in df.iterrows():
    # Extract patient utterance text
    patient_text = str(row.get('utterances', row.get('description', '')))
    
    if not patient_text.strip():
        continue

    response = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': patient_text}
        ]
    )
    
    generated_text = response['message']['content']
    
    results.append({
        'id': row.get('id', idx),
        'patient_input': patient_text,
        'refined_information_score': row.get('refined_information_score', 0),
        'llama_response': generated_text
    })
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
        print(f"Processed {idx + 1}/{len(df)} conversations...")

# 5. Save results
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nExecution complete! Output saved to: {OUTPUT_FILE}")