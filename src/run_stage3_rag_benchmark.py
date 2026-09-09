import json
import chromadb
import ollama
import pandas as pd
from pathlib import Path

# Paths
GRAPH_FILE = Path("data/results/dual_graph_indices.json")
OUTPUT_RAG_FILE = Path("data/results/rag_benchmark_outputs.csv")

# Load Dual-Graph data
with open(GRAPH_FILE, 'r') as f:
    graph_db = json.load(f)

# 1. Initialize ChromaDB (Vector RAG Baseline)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="patient_conversations")

# Populate Vector DB
for idx, item in enumerate(graph_db[:50]): # Benchmark on subset for speed
    collection.add(
        documents=[item['patient_input']],
        ids=[str(item['id'])],
        metadatas=[{"affect_nodes": ", ".join(item['affective_graph_nodes'])}]
    )

print("ChromaDB Vector Store populated.")

# 2. Benchmark Loop
rag_results = []
SYSTEM_PROMPT = "You are a clinical AI. Provide a compassionate and medically accurate response summarizing the patient's concerns and outline next steps."

for item in graph_db[:20]: # Run 20 test comparisons
    patient_text = item['patient_input']
    
    # --- Standard Vector RAG Context ---
    vector_res = collection.query(query_texts=[patient_text], n_results=1)
    vector_context = vector_res['documents'][0][0] if vector_res['documents'] else ""
    
    # --- Dual-GraphRAG Context (Augmenting Vector Context with Explicit Affect Nodes) ---
    affect_nodes = item['affective_graph_nodes']
    clinical_nodes = item['clinical_graph_nodes']
    
    graph_context = (
        f"PATIENT CONTEXT: {patient_text}\n"
        f"DETECTED AFFECTIVE/EMOTIONAL NODES: {', '.join(affect_nodes)}\n"
        f"DETECTED CLINICAL NODES: {', '.join(clinical_nodes)}"
    )

    # Generate Output: Standard Vector RAG
    res_vector = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Context: {vector_context}\nAnswer the patient."}
        ]
    )

    # Generate Output: Dual-GraphRAG
    res_graph = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Context: {graph_context}\nAnswer the patient while addressing both clinical and affective nodes explicitly."}
        ]
    )

    rag_results.append({
        'id': item['id'],
        'patient_input': patient_text,
        'vector_rag_output': res_vector['message']['content'],
        'dual_graphrag_output': res_graph['message']['content']
    })

# Save Benchmark Results
df_rag = pd.DataFrame(rag_results)
df_rag.to_csv(OUTPUT_RAG_FILE, index=False)
print(f"\nRAG Benchmark execution complete! Saved to {OUTPUT_RAG_FILE}")