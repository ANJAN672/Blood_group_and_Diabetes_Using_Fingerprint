"""Evaluate a trained model on a labeled image directory and save probabilities + metrics.

Usage:
  F:/diabetes/.venv/Scripts/python.exe evaluate.py --data-dir dataset_labeled --model models/diabetes_model.h5 --out results.csv

The script expects the dataset folder to be structured like:
  dataset_labeled/
    diabetic/
    non_diabetic/

It writes a CSV with columns: path, true_label, prob, pred_label
and prints metrics and saves a confusion matrix and calibration plot to the current folder.
"""
import os
import argparse
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image

def load_image(path, target_size=(224,224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def iter_image_files(data_dir):
    p = Path(data_dir)
    classes = [d.name for d in p.iterdir() if d.is_dir()]
    classes.sort()
    for cls in classes:
        for f in (p/cls).iterdir():
            if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff') and f.is_file():
                yield str(f), cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', default='models/diabetes_model.h5')
    parser.add_argument('--out', default='predictions_eval.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # lazy import heavy libs
    import tensorflow as tf
    from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, brier_score_loss)
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if not os.path.exists(args.model):
        raise SystemExit('Model file not found: ' + args.model)
    if not os.path.exists(args.data_dir):
        raise SystemExit('Data dir not found: ' + args.data_dir)

    model = tf.keras.models.load_model(args.model)

    rows = []
    y_true = []
    y_prob = []

    for path, label in iter_image_files(args.data_dir):
        try:
            x = load_image(path)
            pred = model.predict(np.expand_dims(x, axis=0))[0][0]
            p = float(pred)
            # interpret according to class_names.json ordering if present
            try:
                class_file = 'models/class_names.json'
                if os.path.exists(class_file):
                    import json
                    cn = json.load(open(class_file,'r'))
                else:
                    cn = None
            except Exception:
                cn = None

            prob = p
            try:
                if cn and len(cn) >= 2 and cn[1] != 'diabetic':
                    prob = 1.0 - p
            except Exception:
                prob = p
            y_prob.append(prob)
            true = 1 if label == 'diabetic' else 0
            y_true.append(true)
            pred_label = 'diabetic' if prob >= 0.5 else 'non_diabetic'
            rows.append([path, label, f'{prob:.6f}', pred_label])
        except Exception as e:
            print('Error processing', path, e)

    # write CSV
    with open(args.out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['path','true_label','prob','pred_label'])
        for r in rows:
            writer.writerow(r)

    if len(y_true) == 0:
        print('No images processed.')
        return

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # metrics
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    print('Evaluation results:')
    print(f'  samples: {len(y_true)}')
    print(f'  AUC: {auc:.4f}')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall: {rec:.4f}')
    print(f'  F1: {f1:.4f}')
    print(f'  Brier score: {brier:.6f}')

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix (rows=true, cols=pred):')
    print(cm)

    # ROC curve data saved as plot
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curve.png')
        print('Saved roc_curve.png')
    except Exception as e:
        print('Could not save ROC curve:', e)

    # calibration plot
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(figsize=(6,6))
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration curve')
        plt.grid(True)
        plt.savefig('calibration.png')
        print('Saved calibration.png')
    except Exception as e:
        print('Could not save calibration plot:', e)


if __name__ == '__main__':
    main()
