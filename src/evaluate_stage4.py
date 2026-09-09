import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
INPUT_FILE = Path("data/results/model_baseline_comparisons.csv")
METRICS_FILE = Path("data/results/adr_berr_metrics.csv")
PLOT_FILE = Path("data/visualizations/epistemic_erasure_benchmark.png")
PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Lexicons
AFFECT_WORDS = [
    "fear", "afraid", "terrified", "scared", "worried", "worry", "anxiety", 
    "anxious", "distress", "distressed", "panic", "panicked", "hopeless", 
    "depressed", "overwhelmed", "painful", "unbearable", "crying", "agony", "scared"
]

CLINICAL_WORDS = [
    "fever", "cough", "pain", "infection", "headache", "symptom", "diagnosis", 
    "test", "positive", "negative", "doctor", "hospital", "x-ray", "treatment", 
    "medication", "blood", "pharyngitis", "thrush", "throat", "lung", "scarring"
]

def count_matches(text, lexicon):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) for w in lexicon)

# Calculate word counts across original input vs. model summaries
df['input_affect'] = df['patient_input'].apply(lambda x: count_matches(x, AFFECT_WORDS))
df['llama_affect'] = df['llama3_2_output'].apply(lambda x: count_matches(x, AFFECT_WORDS))

df['input_clinical'] = df['patient_input'].apply(lambda x: count_matches(x, CLINICAL_WORDS))
df['llama_clinical'] = df['llama3_2_output'].apply(lambda x: count_matches(x, CLINICAL_WORDS))

# Define ADR and BERR
# ADR = % loss of affect
# BERR = % retention of medical entities
def calc_rate(orig, gen, mode="loss"):
    if orig == 0:
        return 0.0
    if mode == "loss":
        return max(0.0, ((orig - gen) / orig) * 100)
    else: # retention
        return min(100.0, (gen / orig) * 100)

df['ADR'] = df.apply(lambda r: calc_rate(r['input_affect'], r['llama_affect'], mode="loss"), axis=1)
df['BERR'] = df.apply(lambda r: calc_rate(r['input_clinical'], r['llama_clinical'], mode="retention"), axis=1)

# Print Summary Results
mean_adr = df['ADR'].mean()
mean_berr = df['BERR'].mean()

print("=" * 60)
print("EPISTEMIC ERASURE AUDIT RESULTS")
print("=" * 60)
print(f"Mean Affective Drop-off Rate (ADR):          {mean_adr:.2f}%")
print(f"Mean Biomedical Entity Retention Rate (BERR): {mean_berr:.2f}%")
print("=" * 60)

# Save computed metrics
df.to_csv(METRICS_FILE, index=False)

# Create Visualizations
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Comparison Bar Chart: Affect Loss vs. Entity Retention
categories = ['Biomedical Entity Retention Rate (BERR)', 'Affective Drop-off Rate (ADR)']
scores = [mean_berr, mean_adr]
colors = ['#2b5c8f', '#d9534f']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, scores, color=colors, width=0.5)

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Epistemic Erasure Benchmark: LLaMA-3.2 Baseline Audit', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)

# Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
print(f"Visualization saved successfully to: {PLOT_FILE}")