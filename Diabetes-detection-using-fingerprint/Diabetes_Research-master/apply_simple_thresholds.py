"""Apply simple probability thresholds to predictions and save the assigned labels.

Rules (0..1):
 - p <= 0.80 -> not_diabetic
 - 0.80 < p <= 0.85 -> not_likely
 - 0.85 < p <= 0.90 -> likely
 - 0.90 < p <= 0.95 -> diabetic
 - p > 0.95 -> highly_diabetic

Saves: predictions_thresholded.csv
Prints counts per category.
"""
import pandas as pd
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description='Apply simple thresholds to predictions CSV')
parser.add_argument('--in', dest='infile', default='predictions_eval.csv', help='Input predictions CSV path')
args = parser.parse_args()

IN = Path(args.infile)
OUT = Path('predictions_thresholded.csv')

if not IN.exists():
    raise SystemExit('predictions_eval.csv not found in repo root')

df = pd.read_csv(IN)

def assign_label(p):
    try:
        p = float(p)
    except Exception:
        return 'unknown'
    if p <= 0.80:
        return 'not_diabetic'
    if p <= 0.85:
        return 'not_likely'
    if p <= 0.90:
        return 'likely'
    if p <= 0.95:
        return 'diabetic'
    return 'highly_diabetic'

df['threshold_label'] = df['prob'].apply(assign_label)
df.to_csv(OUT, index=False)

print('Wrote', OUT)
print('\nCounts by threshold_label:')
print(df['threshold_label'].value_counts())

print('\nSample rows:')
print(df[['path','prob','pred_label','threshold_label']].head(10).to_string(index=False))
