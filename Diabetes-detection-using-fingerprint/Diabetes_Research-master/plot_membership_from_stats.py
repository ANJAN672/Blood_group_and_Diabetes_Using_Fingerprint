"""Plot triangular fuzzy membership functions from summary statistics.

This script uses the provided counts and medians per fuzzy label to place
triangular membership functions on the 0..1 probability axis. Bases are set
midway between adjacent medians; edge bases extend to 0 and 1. For an empty
category (count==0) we place its center slightly above the previous non-empty
center.

Saves: membership_from_summary.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Use fixed thresholds provided now: 0.4, 0.6, 0.7, 0.85.
# We'll create 5 triangular MFs corresponding to the five linguistic labels.
# Ranges (inclusive/exclusive boundaries):
# not_diabetic: p <= 0.40
# not_likely:  0.40 < p <= 0.60
# likely:      0.60 < p <= 0.70
# diabetic:    0.70 < p <= 0.85
# highly_diabetic: p > 0.85

labels = ['not_diabetic', 'not_likely', 'likely', 'diabetic', 'highly_diabetic']
boundaries = [0.0, 0.40, 0.60, 0.70, 0.85, 1.0]


# For each label i, left base = boundaries[i], right base = boundaries[i+1]
bases_left = [boundaries[i] for i in range(len(labels))]
bases_right = [boundaries[i+1] for i in range(len(labels))]
# For trapezoids: flat top between inner 1/3 and 2/3 of the interval
def trap_params(a, d):
    # a = left base, d = right base
    width = d - a
    b = a + width/3
    c = d - width/3
    return a, b, c, d
traps = [trap_params(bases_left[i], bases_right[i]) for i in range(len(labels))]

# Prepare x axis
x = np.linspace(0.0, 1.0, 1000)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function with base [a,d], flat top [b,c]."""
    res = np.zeros_like(x)
    # rising edge
    left = (x >= a) & (x < b)
    if b != a:
        res[left] = (x[left] - a) / (b - a)
    else:
        res[left] = 1.0
    # flat top
    mid = (x >= b) & (x <= c)
    res[mid] = 1.0
    # falling edge
    right = (x > c) & (x <= d)
    if d != c:
        res[right] = (d - x[right]) / (d - c)
    else:
        res[right] = 1.0
    return np.clip(res, 0.0, 1.0)

# Colors

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.figure(figsize=(10,5))
for i, lab in enumerate(labels):
    a, b, c, d = traps[i]
    y = trapmf(x, a, b, c, d)
    plt.plot(x, y, label=f"{lab}", color=colors[i], linewidth=2)
    # mark flat top
    plt.axvline(b, color=colors[i], linestyle='--', alpha=0.7)
    plt.axvline(c, color=colors[i], linestyle='--', alpha=0.7)
    plt.text((b+c)/2, 0.02 + 0.02*i, f"{b:.2f}-{c:.2f}", color=colors[i], rotation=0, va='bottom', ha='center')

plt.title('Triangular membership functions derived from summary medians')
plt.xlabel('Neural model probability (0..1)')
plt.ylabel('Membership degree')
plt.ylim(-0.02, 1.05)
plt.xlim(0.0, 1.0)
plt.grid(alpha=0.2)
plt.legend(loc='upper left')

OUT = Path('membership_from_summary.png')
plt.tight_layout()
plt.savefig(OUT, dpi=150)
plt.close()
print('Saved membership plot to', OUT)
