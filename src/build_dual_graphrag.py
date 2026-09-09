import pandas as pd
import networkx as nx
import json
import re
from pathlib import Path

# Paths
INPUT_FILE = Path("data/processed/final_300_conversations.csv")
OUTPUT_GRAPH_FILE = Path("data/results/dual_graph_indices.json")
OUTPUT_GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Lexicons for Graph Node Extraction
AFFECT_KEYWORDS = [
    "fear", "afraid", "terrified", "scared", "worried", "worry", 
    "anxiety", "anxious", "distress", "distressed", "panic", "panicked", 
    "hopeless", "depressed", "overwhelmed", "painful", "unbearable", "crying"
]

CLINICAL_KEYWORDS = [
    "fever", "cough", "pain", "infection", "headache", "symptom", "diagnosis", 
    "test", "positive", "negative", "x-ray", "treatment", "medication", "blood"
]

graph_database = []

print("Constructing Dual-Graph Indices (Clinical Subgraph + Affective Subgraph)...")

for idx, row in df.iterrows():
    text = str(row.get('utterances', row.get('description', ''))).lower()
    
    # Extract Graph Nodes
    affect_nodes = [word for word in AFFECT_KEYWORDS if re.search(r'\b' + re.escape(word) + r'\b', text)]
    clinical_nodes = [word for word in CLINICAL_KEYWORDS if re.search(r'\b' + re.escape(word) + r'\b', text)]
    
    # Build NetworkX Dual-Graph Structures
    G_clinical = nx.Graph()
    G_affective = nx.Graph()
    
    # Add Clinical Edges
    for i in range(len(clinical_nodes)):
        for j in range(i + 1, len(clinical_nodes)):
            G_clinical.add_edge(clinical_nodes[i], clinical_nodes[j], relation="co_occurrence")
            
    # Add Affective Edges & Cross-Links (Linking Affect to Clinical Symptoms)
    for affect in affect_nodes:
        for clinical in clinical_nodes:
            G_affective.add_edge(affect, clinical, relation="patient_experience")

    graph_database.append({
        'id': row.get('id', idx),
        'patient_input': row.get('utterances', row.get('description', '')),
        'clinical_graph_nodes': list(G_clinical.nodes()),
        'affective_graph_nodes': list(G_affective.nodes()),
        'dual_edges': list(G_affective.edges())
    })

# Save structured graph indices
with open(OUTPUT_GRAPH_FILE, 'w') as f:
    json.dump(graph_database, f, indent=4)

print(f"\nDual-Graph indexing complete! Saved to {OUTPUT_GRAPH_FILE}")