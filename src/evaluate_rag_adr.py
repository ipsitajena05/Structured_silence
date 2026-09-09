import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("data/results/rag_benchmark_outputs.csv")
df = pd.read_csv(INPUT_FILE)

AFFECT_KEYWORDS = [
    "fear", "afraid", "terrified", "scared", "worried", "worry", 
    "anxiety", "anxious", "distress", "distressed", "panic", "panicked", 
    "hopeless", "depressed", "overwhelmed", "painful", "unbearable", "crying"
]

def count_matches(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) for w in AFFECT_KEYWORDS)

df['input_affect'] = df['patient_input'].apply(count_matches)
df['vector_affect'] = df['vector_rag_output'].apply(count_matches)
df['graph_affect'] = df['dual_graphrag_output'].apply(count_matches)

def calc_adr(orig, gen):
    if orig == 0:
        return 0.0
    return max(0.0, ((orig - gen) / orig) * 100)

df['vector_adr'] = df.apply(lambda r: calc_adr(r['input_affect'], r['vector_affect']), axis=1)
df['graph_adr'] = df.apply(lambda r: calc_adr(r['input_affect'], r['graph_affect']), axis=1)

print("=" * 60)
print("FINAL RAG BENCHMARK RESULTS")
print("=" * 60)
print(f"Standard Vector RAG Average ADR: {df['vector_adr'].mean():.2f}%")
print(f"Dual-GraphRAG Average ADR:        {df['graph_adr'].mean():.2f}%")
print("=" * 60)