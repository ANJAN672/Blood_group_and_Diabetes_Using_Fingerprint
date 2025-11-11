"""Extract interpretable features from fingerprint images for fuzzy modeling.

Outputs a CSV `features.csv` with one row per image and columns:
 - path, true_label
 - mean_intensity, std_intensity, entropy
 - gradient_mean, edge_density
 - ridge_density (proportion of dark pixels after binarization)
 - orientation_mean (radians, in [-pi,pi])
 - skeleton_density (if skimage available, else empty)
 - minutiae_endpoints (if skeleton available, else empty)

Usage:
  F:/diabetes/.venv/Scripts/python.exe extract_features.py --data-dir dataset_labeled --out features.csv

The script tries to avoid heavy dependencies; skeletonization is attempted only if scikit-image is present.
"""
import argparse
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image


def load_gray(path, size=(224,224)):
    img = Image.open(path).convert('L')
    img = img.resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return arr


def entropy_from_hist(img):
    # img assumed 0-255
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,255))
    probs = hist / (hist.sum() + 1e-12)
    probs = probs[probs>0]
    return -np.sum(probs * np.log2(probs))


def gradient_stats(img):
    # simple sobel-like gradients using finite differences
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:,1:-1] = (img[:,2:] - img[:,:-2]) / 2.0
    gy[1:-1,:] = (img[2:,:] - img[:-2,:]) / 2.0
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return mag.mean(), ang


def edge_density_from_mag(mag, thresh=None):
    if thresh is None:
        thresh = np.mean(mag) * 0.8
    edges = mag > thresh
    return float(edges.mean())


def binarize_otsu(img):
    # lightweight Otsu implementation
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0,255))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    threshold = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    bw = img <= threshold
    return bw.astype(np.uint8), threshold


def skeleton_and_minutiae(binary_img):
    # binary_img: 0/1 uint8, foreground assumed 1 (ridges)
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(binary_img > 0)
        sk = sk.astype(np.uint8)
        # compute neighbor counts via shifts
        neigh = np.zeros_like(sk, dtype=np.uint8)
        shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dy,dx in shifts:
            neigh += np.roll(np.roll(sk, dy, axis=0), dx, axis=1)
        # endpoints: pixels with exactly 1 neighbor
        endpoints = ((sk==1) & (neigh==1)).sum()
        branchpoints = ((sk==1) & (neigh>=3)).sum()
        return float(sk.mean()), int(endpoints), int(branchpoints)
    except Exception:
        return None, None, None


def circular_mean_angle(angle_array):
    # angles in radians
    a = np.array(angle_array).flatten()
    sin_sum = np.sum(np.sin(a))
    cos_sum = np.sum(np.cos(a))
    return math.atan2(sin_sum, cos_sum)


def process_image(path):
    img = load_gray(path)
    mean_i = float(img.mean()/255.0)
    std_i = float(img.std()/255.0)
    ent = float(entropy_from_hist(img))
    grad_mean, ang = gradient_stats(img)
    edge_den = edge_density_from_mag(np.hypot(*(np.gradient(img))))
    bw, thr = binarize_otsu(img)
    ridge_density = float((bw>0).mean())
    sk_density, endpoints, branchpoints = skeleton_and_minutiae(bw)
    orient_mean = float(circular_mean_angle(ang))
    return {
        'mean_intensity': mean_i,
        'std_intensity': std_i,
        'entropy': ent,
        'gradient_mean': float(grad_mean),
        'edge_density': float(edge_den),
        'ridge_density': ridge_density,
        'skeleton_density': sk_density,
        'minutiae_endpoints': endpoints,
        'minutiae_branchpoints': branchpoints,
        'binarize_threshold': int(thr),
        'orientation_mean': orient_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset_labeled')
    parser.add_argument('--out', default='features.csv')
    args = parser.parse_args()

    p = Path(args.data_dir)
    if not p.exists():
        raise SystemExit('Data dir not found: ' + str(p))

    rows = []
    for cls in sorted([d.name for d in p.iterdir() if d.is_dir()]):
        for f in (p/cls).iterdir():
            if f.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
                continue
            feats = process_image(str(f))
            row = {'path': str(f), 'label': cls}
            row.update(feats)
            rows.append(row)
            print('Processed', f.name)

    # write CSV
    keys = ['path','label','mean_intensity','std_intensity','entropy','gradient_mean','edge_density',
            'ridge_density','skeleton_density','minutiae_endpoints','minutiae_branchpoints','binarize_threshold','orientation_mean']
    with open(args.out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([r.get(k,'') for k in keys])
    print('Wrote features to', args.out)

if __name__ == '__main__':
    import argparse
    main()
