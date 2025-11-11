import os
import argparse
import csv
from PIL import Image
import numpy as np
import tensorflow as tf


def load_image(path, target_size=(224,224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_folder(model_path, folder, out_csv='predictions.csv', max_files=None):
    if not os.path.exists(model_path):
        raise SystemExit('Model not found: ' + model_path)
    model = tf.keras.models.load_model(model_path)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
    files.sort()
    if max_files:
        files = files[:max_files]

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['path','probability','label'])
        for f in files:
            try:
                x = load_image(f)
                pred = model.predict(np.expand_dims(x, axis=0))[0][0]
                prob = float(pred) * 100.0
                label = 'diabetic' if pred >= 0.5 else 'non_diabetic'
                writer.writerow([f, f'{prob:.4f}', label])
            except Exception as e:
                writer.writerow([f, 'error', str(e)])

    print('Wrote predictions to', out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/diabetes_model.h5')
    parser.add_argument('--folder', required=True)
    parser.add_argument('--out', default='predictions.csv')
    parser.add_argument('--max', type=int, default=20)
    args = parser.parse_args()
    predict_folder(args.model, args.folder, args.out, args.max)


if __name__ == '__main__':
    main()
