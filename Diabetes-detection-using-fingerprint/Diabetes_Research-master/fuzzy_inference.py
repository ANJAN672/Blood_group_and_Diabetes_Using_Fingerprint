"""
Simple Mamdani fuzzy inference system for diabetes risk using three features:
 - edge_density
 - ridge_density
 - entropy

Steps:
 - Read `features.csv` and compute 10/50/90 percentiles per feature.
 - Define triangular membership functions (low/med/high) using those percentiles.
 - Define a set of intuitive rules mapping combinations to one of five outputs:
    not_diabetic, not_likely, likely, diabetic, highly_diabetic
 - For each sample, compute rule firing strengths, aggregate output memberships, defuzzify by centroid.
 - Save `fuzzy_outputs.csv` with columns: path,label,fuzzy_score,assigned_label

This is a simple, interpretable prototype to compare with the neural model.
"""
import csv
from pathlib import Path
import numpy as np

FEATURES_CSV = Path('features.csv')
OUT_CSV = Path('fuzzy_outputs.csv')

if not FEATURES_CSV.exists():
    raise SystemExit('features.csv not found. Run extract_features.py first')

# read data
rows = []
with open(FEATURES_CSV,'r',newline='') as fh:
    reader = csv.DictReader(fh)
    fields = reader.fieldnames
    for r in reader:
        rows.append(r)

# features we will use
feat_names = ['edge_density','ridge_density','entropy']
# collect arrays
feat_vals = {f: [] for f in feat_names}
for r in rows:
    for f in feat_names:
        try:
            feat_vals[f].append(float(r.get(f,'')))
        except Exception:
            feat_vals[f].append(np.nan)
for f in feat_names:
    feat_vals[f] = np.array(feat_vals[f], dtype=float)
    # replace nan with median
    if np.isnan(feat_vals[f]).any():
        m = np.nanmedian(feat_vals[f])
        feat_vals[f][np.isnan(feat_vals[f])] = m

# compute percentiles
p10 = {f: float(np.percentile(feat_vals[f],10)) for f in feat_names}
p50 = {f: float(np.percentile(feat_vals[f],50)) for f in feat_names}
p90 = {f: float(np.percentile(feat_vals[f],90)) for f in feat_names}

# triangular membership
def tri(x,a,b,c):
    x = np.array(x, dtype=float)
    left = np.where(x<=b, (x-a)/(b-a+1e-12), (c-x)/(c-b+1e-12))
    return np.clip(left, 0.0, 1.0)

# output domain (0..1) and output MFs (triangular centered at 0,0.25,0.5,0.75,1)
out_centers = {'not_diabetic':0.0, 'not_likely':0.25, 'likely':0.5, 'diabetic':0.75, 'highly_diabetic':1.0}
# triangular output MFs defined so adjacent overlap
out_names = list(out_centers.keys())

# helper to compute output membership function values for a clipped strength (Mamdani: clip MF)
def output_mf_values(domain, center):
    # triangles where (a,b,c) are center spacing
    # we choose spacing 0.25 between centers, so for center c: a=c-0.25, b=c, c=c+0.25
    a = max(0.0, center - 0.25)
    b = center
    c = min(1.0, center + 0.25)
    return tri(domain, a, b, c)

# define rules: tuple of conditions -> output_label
# each condition is (feature, term) where term in {'low','med','high'}; multiple conditions combined with AND (min)
# We create intuitive rules:
rules = [
    # strong evidence of diabetic
    ([('edge_density','high'), ('ridge_density','high'), ('entropy','high')], 'highly_diabetic'),
    ([('edge_density','high'), ('ridge_density','high')], 'diabetic'),
    ([('edge_density','high'), ('entropy','high')], 'diabetic'),
    ([('ridge_density','high'), ('entropy','high')], 'diabetic'),

    # likely
    ([('edge_density','med'), ('ridge_density','high')], 'likely'),
    ([('edge_density','high'), ('ridge_density','med')], 'likely'),
    ([('edge_density','med'), ('entropy','med')], 'likely'),

    # not likely
    ([('edge_density','low'), ('ridge_density','med')], 'not_likely'),
    ([('edge_density','med'), ('ridge_density','low')], 'not_likely'),

    # clear non-diabetic
    ([('edge_density','low'), ('ridge_density','low'), ('entropy','low')], 'not_diabetic'),
    ([('entropy','low')], 'not_diabetic'),
]

# evaluate each sample
out_rows = []
# domain for defuzzification
domain = np.linspace(0.0, 1.0, 201)

for idx, r in enumerate(rows):
    # get feature values
    vals = {f: float(r.get(f,0.0)) for f in feat_names}
    # compute membership for inputs
    mem = {}
    for f in feat_names:
        x = vals[f]
        mem.setdefault(f, {})
        mem[f]['low'] = float(tri(x, p10[f], p10[f], p50[f]))
        mem[f]['med'] = float(tri(x, p10[f], p50[f], p90[f]))
        mem[f]['high'] = float(tri(x, p50[f], p90[f], p90[f]))
    # rule evaluation
    # aggregated output membership (initially zero)
    agg = {name: np.zeros_like(domain) for name in out_names}
    # keep track of rule strengths for debugging
    rule_strs = []
    for conds, out_label in rules:
        # compute firing strength (min of memberships for all conds)
        strengths = []
        for feat, term in conds:
            strengths.append(mem[feat][term])
        firing = float(min(strengths)) if strengths else 0.0
        if firing <= 0: 
            continue
        rule_strs.append((conds, out_label, firing))
        # form the clipped output MF (Mamdani): clip the output MF at firing
        out_center = out_centers[out_label]
        mf = output_mf_values(domain, out_center)
        clipped = np.minimum(mf, firing)
        agg[out_label] = np.maximum(agg[out_label], clipped)

    # aggregate all outputs into a single aggregated MF (max across labels)
    combined = np.zeros_like(domain)
    for lbl in out_names:
        combined = np.maximum(combined, agg[lbl])

    # defuzzify by centroid
    if combined.sum() == 0:
        fuzzy_score = 0.0
    else:
        fuzzy_score = float(np.sum(domain * combined) / (np.sum(combined)+1e-12))

    # map fuzzy_score to a discrete label by nearest center (or thresholds)
    # thresholds: <0.2 not_diabetic, 0.2-0.4 not_likely, 0.4-0.6 likely, 0.6-0.8 diabetic, >=0.8 highly
    if fuzzy_score < 0.2:
        assigned = 'not_diabetic'
    elif fuzzy_score < 0.4:
        assigned = 'not_likely'
    elif fuzzy_score < 0.6:
        assigned = 'likely'
    elif fuzzy_score < 0.8:
        assigned = 'diabetic'
    else:
        assigned = 'highly_diabetic'

    out_rows.append({'path': r['path'], 'true_label': r['label'], 'fuzzy_score': fuzzy_score, 'assigned_label': assigned})

# write CSV
with open(OUT_CSV, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['path','true_label','fuzzy_score','assigned_label'])
    for o in out_rows:
        writer.writerow([o['path'], o['true_label'], f"{o['fuzzy_score']:.6f}", o['assigned_label']])

# print summary counts
from collections import Counter
cnt = Counter([o['assigned_label'] for o in out_rows])
print('Fuzzy label counts:')
for k,v in cnt.items():
    print(f'  {k}: {v}')

print('\nWrote fuzzy outputs to', OUT_CSV)
