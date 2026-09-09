import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set output path
OUTPUT_PLOT = Path("data/visualizations/final_rag_adr_comparison.png")
OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

# Data
methods = ['Standard Vector RAG', 'Dual-GraphRAG (Ours)']
adr_scores = [5.00, 0.00]
colors = ['#d9534f', '#2b5c8f']

# Set style
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars
bars = ax.bar(methods, adr_scores, color=colors, width=0.4)

# Formatting
ax.set_ylabel('Affective Drop-off Rate (ADR %)', fontsize=12, fontweight='bold')
ax.set_title('Stage 04 Evaluation: Eradicating Epistemic Erasure via Dual-GraphRAG', fontsize=13, fontweight='bold')
ax.set_ylim(0, 10)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0, 
        yval + 0.3, 
        f"{yval:.2f}%", 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"Final plot successfully saved to: {OUTPUT_PLOT}")