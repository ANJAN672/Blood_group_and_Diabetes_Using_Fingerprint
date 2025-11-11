"""Create a single combined figure containing membership plots for all numeric features.

Saves `membership_all_features.png` in the repository root.
"""
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_CSV = Path('features.csv')
OUT_PATH = Path('membership_all_features.png')

if not FEATURES_CSV.exists():
    raise SystemExit('features.csv not found. Run extract_features.py first.')

# load data
rows = []
with open(FEATURES_CSV, 'r', newline='') as fh:
    reader = csv.DictReader(fh)
    fields = reader.fieldnames
    for r in reader:
        rows.append(r)

numeric_fields = [f for f in fields if f not in ('path','label')]
# gather arrays
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

# triangular membership
def tri(x, a, b, c):
    left = np.where(x<=b, (x-a)/(b-a+1e-12), (c-x)/(c-b+1e-12))
    return np.clip(left, 0.0, 1.0)

# layout: try to create a compact grid
num = len(data)
cols = 3
rowsn = math.ceil(num/cols)
fig, axes = plt.subplots(rowsn, cols, figsize=(4*cols, 3*rowsn))
axes = axes.flatten()

for i, (f, arr) in enumerate(data.items()):
    ax = axes[i]
    if arr.size == 0:
        ax.text(0.5, 0.5, f + '\n(no numeric data)', ha='center', va='center')
        ax.axis('off')
        continue
    lo = np.percentile(arr, 10)
    mid = np.percentile(arr, 50)
    hi = np.percentile(arr, 90)
    rng_low = lo - 0.05*abs(lo if lo!=0 else 1)
    rng_high = hi + 0.05*abs(hi if hi!=0 else 1)
    rng = np.linspace(rng_low, rng_high, 300)

    m_low = tri(rng, lo, lo, mid)
    m_med = tri(rng, lo, mid, hi)
    m_high = tri(rng, mid, hi, hi)

    ax.plot(rng, m_low, label='Low', color='C0')
    ax.plot(rng, m_med, label='Medium', color='C1')
    ax.plot(rng, m_high, label='High', color='C2')
    ax.fill_between(rng, 0, m_med, color='C1', alpha=0.05)
    ax.set_title(f)
    ax.set_xlabel(f)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # histogram on secondary axis (smaller, transparent)
    ax2 = ax.twinx()
    ax2.hist(arr, bins=12, density=True, alpha=0.4, color='C3')
    ax2.set_ylabel('Density')

    # vertical lines for peaks
    ax.axvline(lo, color='C0', linestyle='--', alpha=0.6)
    ax.axvline(mid, color='C1', linestyle='--', alpha=0.6)
    ax.axvline(hi, color='C2', linestyle='--', alpha=0.6)

    # small legend
    ax.legend(loc='upper right', fontsize='small')

# hide any leftover axes
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Fuzzy membership functions (Low/Medium/High) for all features', fontsize=14)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_PATH, dpi=160)
plt.close()
print('Saved combined membership figure to', OUT_PATH)
