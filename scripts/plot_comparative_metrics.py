import matplotlib.pyplot as plt
import pandas as pd
import os

# Provided metrics
metrics = {
    'Emoji Model': {'AUC': 0.6618, 'Precision@5': 0.0288, 'Recall@5': 0.0647},
    'TF-IDF Model': {'AUC': 0.4181, 'Precision@5': 0.0011, 'Recall@5': 0.0025},
    'LDA Model': {'AUC': 0.6472, 'Precision@5': 0.0318, 'Recall@5': 0.0698},
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics).T

# Print summary table
print("\nComparative Metrics Table:")
print(metrics_df)

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(8, 6))
metrics_df.plot(kind='bar', ax=ax, rot=0, width=0.7)
ax.set_title('Comparative Analysis of Recommendation Models')
ax.set_ylabel('Score')
ax.set_xlabel('Model')
ax.legend(title='Metric')
plt.tight_layout()

# Save plot
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'comparative_metrics.png'))
plt.show() 