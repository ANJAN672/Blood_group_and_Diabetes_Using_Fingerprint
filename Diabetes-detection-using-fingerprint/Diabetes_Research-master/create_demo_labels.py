import os
import csv
from glob import glob

def main():
    os.makedirs('labels', exist_ok=True)
    files = sorted(glob(os.path.join('dataset', '**', '*.*'), recursive=True))
    imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    if not imgs:
        print('No images found under dataset. Ensure dataset is present.')
        return
    sel = imgs[:40]
    n = len(sel)
    with open('labels/diabetes_labels.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['path', 'label'])
        for i, f in enumerate(sel):
            label = 'non_diabetic' if i < n//2 else 'diabetic'
            writer.writerow([f, label])
    print(f'Wrote {len(sel)} demo labels to labels/diabetes_labels.csv')

if __name__ == '__main__':
    main()
