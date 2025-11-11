"""Plot comparison graphs between fuzzy outputs and neural model probabilities.
Creates a single PNG with three panels:
 - Bar chart of fuzzy label counts
 - Scatter: fuzzy_score vs model probability (color by true_label)
 - Boxplot: model probability per fuzzy assigned label

Requires `fuzzy_outputs.csv` and `predictions_eval.csv` to exist.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FUZZY = Path('fuzzy_outputs.csv')
PRED = Path('predictions_eval.csv')
OUT = Path('fuzzy_vs_neural_plots.png')

if not FUZZY.exists() or not PRED.exists():
    raise SystemExit('Required CSVs not found. Ensure fuzzy_outputs.csv and predictions_eval.csv exist.')

fdf = pd.read_csv(FUZZY)
pdf = pd.read_csv(PRED)

# Normalize paths to basenames for join
fdf['basename'] = fdf['path'].apply(lambda p: Path(p).name)
pdf['basename'] = pdf['path'].apply(lambda p: Path(p).name)
# Avoid duplicate 'true_label' column name after merge by renaming explicitly
if 'true_label' in fdf.columns:
    fdf = fdf.rename(columns={'true_label': 'true_label_fuzzy'})
if 'true_label' in pdf.columns:
    pdf = pdf.rename(columns={'true_label': 'true_label_neural'})

# Merge keeping neural labels/probability
df = pd.merge(fdf, pdf[['basename','prob','pred_label','true_label_neural']], on='basename', how='left')
# if model prob column is 0..1 or 0..100, handle both
if df['prob'].max() > 1.5:
    # assume percent
    df['model_prob'] = df['prob']/100.0
else:
    df['model_prob'] = df['prob']

# Plotting
labels_order = ['not_diabetic','not_likely','likely','diabetic','highly_diabetic']

fig, axes = plt.subplots(1,3, figsize=(18,5))

# 1. bar counts
counts = df['assigned_label'].value_counts().reindex(labels_order).fillna(0)
axes[0].bar(counts.index, counts.values, color='C0')
axes[0].set_title('Fuzzy assigned label counts')
axes[0].set_xticks(range(len(counts.index)))
axes[0].set_xticklabels(counts.index, rotation=25)
axes[0].set_ylabel('Count')

# 2. scatter fuzzy_score vs model_prob
colors = {'diabetic':'C1','non_diabetic':'C2'}
# Use the neural true label for coloring if available, fall back to fuzzy true label
label_col_for_color = 'true_label_neural' if 'true_label_neural' in df.columns else ('true_label_fuzzy' if 'true_label_fuzzy' in df.columns else None)
if label_col_for_color is not None:
    axes[1].scatter(df['fuzzy_score'], df['model_prob'], c=df[label_col_for_color].map(colors), s=60, alpha=0.8)
else:
    axes[1].scatter(df['fuzzy_score'], df['model_prob'], color='C0', s=60, alpha=0.8)
axes[1].set_xlabel('Fuzzy score (0..1)')
axes[1].set_ylabel('Neural model probability (0..1)')
axes[1].set_title('Fuzzy score vs Neural probability (color=true label)')
# create legend
for t,c in colors.items():
    axes[1].scatter([],[], c=c, label=t)
axes[1].legend()

# 3. boxplot model_prob per fuzzy assigned label
box_data = [df[df['assigned_label']==lab]['model_prob'].dropna().values for lab in labels_order]
axes[2].boxplot(box_data, labels=labels_order, patch_artist=True)
axes[2].set_title('Model probability by fuzzy assigned label')
axes[2].set_ylabel('Neural model probability (0..1)')
axes[2].set_xticklabels(labels_order, rotation=25)

plt.suptitle('Fuzzy vs Neural comparison')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT, dpi=150)
plt.close()
print('Saved comparison plot to', OUT)

# Also print a small table of mean neural prob per fuzzy label
summary = df.groupby('assigned_label')['model_prob'].agg(['count','mean','median'])
print('\nMean neural probability per fuzzy label:')
print(summary.reindex(labels_order).fillna(0))
