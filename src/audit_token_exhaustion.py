import pandas as pd
import json
import tiktoken
from pathlib import Path

# Paths
GRAPH_FILE = Path("data/results/dual_graph_indices.json")
OUTPUT_CSV = Path("data/results/token_exhaustion_metrics.csv")

# Load graph data
with open(GRAPH_FILE, 'r') as f:
    graph_db = json.load(f)

# Use tiktoken / cl100k_base to approximate LLaMA/GPT token counts
enc = tiktoken.get_encoding("cl100k_base")

token_metrics = []

for item in graph_db:
    patient_text = item['patient_input']
    affect_nodes = item['affective_graph_nodes']
    clinical_nodes = item['clinical_graph_nodes']

    # 1. Standard Vector RAG Prompt (Raw text concatenation)
    raw_vector_prompt = f"Context: {patient_text}\nQuestion: Provide a clinical summary and next steps for the patient."
    vector_tokens = len(enc.encode(raw_vector_prompt))

    # 2. Dual-GraphRAG Prompt (Structured compressed nodes)
    graph_prompt = (
        f"PATIENT CONTEXT: {patient_text}\n"
        f"DETECTED AFFECTIVE NODES: {', '.join(affect_nodes)}\n"
        f"DETECTED CLINICAL NODES: {', '.join(clinical_nodes)}\n"
        f"Question: Provide a clinical summary and next steps for the patient."
    )
    graph_tokens = len(enc.encode(graph_prompt))

    # Check BioGPT 1024 Token Window Threshold
    biogpt_limit = 1024
    vector_exhaustion = vector_tokens > biogpt_limit
    
    token_metrics.append({
        'id': item['id'],
        'vector_rag_tokens': vector_tokens,
        'dual_graph_tokens': graph_tokens,
        'biogpt_truncation_risk': vector_exhaustion
    })

df_tokens = pd.DataFrame(token_metrics)
df_tokens.to_csv(OUTPUT_CSV, index=False)

print("=" * 65)
print("TOKEN EXHAUSTION AUDIT COMPLETE")
print("=" * 65)
print(f"Average Vector RAG Token Length:  {df_tokens['vector_rag_tokens'].mean():.1f} tokens")
print(f"Average Dual-Graph Token Length:  {df_tokens['dual_graph_tokens'].mean():.1f} tokens")
print(f"Conversations Exceeding BioGPT 1024 Limit: {df_tokens['biogpt_truncation_risk'].sum()} / {len(df_tokens)}")
print("=" * 65)