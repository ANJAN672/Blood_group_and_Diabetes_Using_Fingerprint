import os
import shutil
import pandas as pd


def apply_labels(csv_path='labels/diabetes_labels.csv', output_dir='dataset_labeled'):
    if not os.path.exists(csv_path):
        raise SystemExit('Labels CSV not found: ' + csv_path)
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    classes = df['label'].unique().tolist()
    for c in classes:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

    for _, row in df.iterrows():
        src = row['path']
        label = row['label']
        if not os.path.exists(src):
            print('Missing file, skipping:', src)
            continue
        dst = os.path.join(output_dir, label, os.path.basename(src))
        shutil.copy2(src, dst)

    print('Labeled dataset created at', output_dir)


if __name__ == '__main__':
    apply_labels()
