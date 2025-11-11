"""Compute basic statistics for features.csv and print suggested fuzzy cut points.
"""
import csv
import numpy as np
from collections import defaultdict

FEATURES = []
rows = []
with open('features.csv','r',newline='') as fh:
    reader = csv.DictReader(fh)
    FEATURES = [f for f in reader.fieldnames if f not in ('path','label')]
    for r in reader:
        rows.append(r)

vals = defaultdict(list)
for r in rows:
    for f in FEATURES:
        v = r.get(f,'')
        try:
            vals[f].append(float(v))
        except Exception:
            # skip non-numeric
            pass

print('Feature statistics (count, min, 10%, 50%, 90%, max, mean)')
for f in FEATURES:
    arr = np.array(vals[f], dtype=float)
    if arr.size == 0:
        print(f, ': no numeric values')
        continue
    p10 = np.percentile(arr,10)
    p50 = np.percentile(arr,50)
    p90 = np.percentile(arr,90)
    print(f'{f}: n={arr.size}, min={arr.min():.6f}, p10={p10:.6f}, median={p50:.6f}, p90={p90:.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}')

# Suggest triangular peaks using percentiles 10/50/90
print('\nSuggested triangular peaks (low center=10%, medium=50%, high=90%)')
for f in FEATURES:
    arr = np.array(vals[f], dtype=float)
    if arr.size == 0:
        continue
    p10 = np.percentile(arr,10)
    p50 = np.percentile(arr,50)
    p90 = np.percentile(arr,90)
    print(f'{f}: low_peak={p10:.6f}, med_peak={p50:.6f}, high_peak={p90:.6f}')
