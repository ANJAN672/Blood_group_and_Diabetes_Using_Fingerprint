"""Plot fuzzy membership functions and overlay dataset feature values.

Usage:
  python plot_membership.py --feature edge_density --features features.csv --out membership_edge_density.png

This creates triangular membership functions for 'low', 'medium', 'high' and overlays the histogram of values.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def tri(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a+1e-12), (c-x)/(c-b+1e-12)))


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x-mu)/sigma)**2)


def load_feature_csv(path, feature):
    vals = []
    with open(path,'r',newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            v = row.get(feature,'')
            try:
                vals.append(float(v))
            except Exception:
                continue
    return np.array(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', required=True)
    parser.add_argument('--features', default='features.csv')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    arr = load_feature_csv(args.features, args.feature)
    if arr.size == 0:
        raise SystemExit('No values found for feature: ' + args.feature)

    lo = arr.min()
    hi = arr.max()
    rng = np.linspace(lo, hi, 400)

    # simple triangular membership: low, medium, high
    a = lo
    c = hi
    b = (lo + hi) / 2.0
    mu1 = lo
    mu2 = b
    mu3 = hi

    m_low = tri(rng, a, a, b)
    m_med = tri(rng, a, b, c)
    m_high = tri(rng, b, c, c)

    plt.figure(figsize=(8,4))
    plt.plot(rng, m_low, label='low')
    plt.plot(rng, m_med, label='medium')
    plt.plot(rng, m_high, label='high')
    plt.fill_between(rng, 0, m_med, color='gray', alpha=0.05)

    # overlay histogram (normalized)
    plt.twinx()
    plt.hist(arr, bins=20, density=True, alpha=0.6, color='C3')
    plt.title('Fuzzy membership for feature: ' + args.feature)
    plt.xlabel(args.feature)
    plt.ylabel('Density')
    plt.legend(loc='upper left')

    out_path = args.out or f'membership_{args.feature}.png'
    plt.savefig(out_path, bbox_inches='tight')
    print('Saved', out_path)

if __name__ == '__main__':
    main()
