"""Generate fuzzy membership plots (low/medium/high) for all numeric features in features.csv.

- Uses percentiles 10/50/90 as the triangular peak centers for low/med/high.
- Saves a PNG per feature named membership_<feature>.png with clear labels and legend.
"""
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

FEATURES_CSV = Path('features.csv')
OUT_DIR = Path('.')

if not FEATURES_CSV.exists():
    raise SystemExit('features.csv not found. Run extract_features.py first.')

# load numeric columns
rows = []
with open(FEATURES_CSV, 'r', newline='') as fh:
    reader = csv.DictReader(fh)
    all_fields = reader.fieldnames
    for r in reader:
        rows.append(r)

numeric_fields = [f for f in all_fields if f not in ('path','label')]

# build arrays
data = {}
for f in numeric_fields:
    vals = []
    for r in rows:
        v = r.get(f,'')
        try:
            vals.append(float(v))
        except Exception:
            pass
    data[f] = np.array(vals)

# helper functions
def tri(x, a, b, c):
    # triangular membership with safe denominators
    left = np.where(x<=b, (x-a)/(b-a+1e-12), (c-x)/(c-b+1e-12))
    return np.clip(left, 0.0, 1.0)

# generate plots
for f, arr in data.items():
    if arr.size == 0:
        print('Skipping', f, '(no numeric data)')
        continue
    lo = np.percentile(arr, 10)
    mid = np.percentile(arr, 50)
    hi = np.percentile(arr, 90)
    rng = np.linspace(min(arr.min(), lo - 0.05*abs(lo if lo!=0 else 1)), max(arr.max(), hi + 0.05*abs(hi if hi!=0 else 1)), 400)

    m_low = tri(rng, lo, lo, mid)
    m_med = tri(rng, lo, mid, hi)
    m_high = tri(rng, mid, hi, hi)

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(rng, m_low, label='Low', color='C0')
    ax1.plot(rng, m_med, label='Medium', color='C1')
    ax1.plot(rng, m_high, label='High', color='C2')
    ax1.set_xlabel(f)
    ax1.set_ylabel('Membership')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # overlay histogram on twin axis
    ax2 = ax1.twinx()
    ax2.hist(arr, bins=20, density=True, alpha=0.4, color='C3')
    ax2.set_ylabel('Density')

    # add vertical lines at peaks
    ax1.axvline(lo, color='C0', linestyle='--', alpha=0.6)
    ax1.axvline(mid, color='C1', linestyle='--', alpha=0.6)
    ax1.axvline(hi, color='C2', linestyle='--', alpha=0.6)

    # neat legend combining axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f'Fuzzy membership (low/med/high) for feature: {f}')
    out = OUT_DIR / f'membership_{f}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Saved', out)

print('Done generating membership plots for', len(data))
