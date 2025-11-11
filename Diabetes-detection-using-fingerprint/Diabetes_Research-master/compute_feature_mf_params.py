"""Compute trapezoidal MF parameters (a,b,c,d) for selected features.

We use data-driven boundaries at percentiles [0,20,40,60,80,100] to create
5 linguistic labels. For each label interval [a,d] we set the flat top to
[b,c] where b = a + (d-a)/3 and c = d - (d-a)/3.

Writes `mf_params_features.json` and prints a compact table.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

IN = Path('features.csv')
OUT = Path('mf_params_features.json')

if not IN.exists():
    raise SystemExit('features.csv not found in repo root')

df = pd.read_csv(IN)

features = ['edge_density', 'ridge_density', 'mean_intensity']
labels = ['not_diabetic','not_likely','likely','diabetic','highly_diabetic']
pct = [0,20,40,60,80,100]

result = {}

for feat in features:
    col = df[feat].dropna().astype(float)
    bounds = np.percentile(col, pct).tolist()
    feat_params = {}
    for i,label in enumerate(labels):
        a = float(bounds[i])
        d = float(bounds[i+1])
        width = d - a
        b = float(a + width/3.0)
        c = float(d - width/3.0)
        feat_params[label] = [round(a,6), round(b,6), round(c,6), round(d,6)]
    result[feat] = {
        'percentiles_used': {str(p): round(float(v),6) for p,v in zip(pct,bounds)},
        'labels': feat_params
    }

with OUT.open('w') as f:
    json.dump(result, f, indent=2)

print('Wrote', OUT)
print()
for feat in features:
    print('Feature:', feat)
    print('Percentiles (0,20,40,60,80,100):')
    for p,v in result[feat]['percentiles_used'].items():
        print('  p'+p+':', v)
    print('Label trapezoid params (a,b,c,d):')
    for lab,vals in result[feat]['labels'].items():
        print(f'  {lab}:', vals)
    print()
