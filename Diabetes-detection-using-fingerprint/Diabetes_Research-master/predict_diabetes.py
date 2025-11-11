"""Predict diabetes probability from fingerprint images using the trained model.

Usage examples:
  # single image
  python predict_diabetes.py --image path\to\fingerprint.png --model models/diabetes_model.h5

  # directory
  python predict_diabetes.py --dir path\to\fingerprints --model models/diabetes_model.h5
"""
import os
import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(path, target_size=(224,224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_single(model, img_path):
    arr = load_image(img_path)
    pred = model.predict(np.expand_dims(arr, axis=0))[0][0]
    # interpret sigmoid output with respect to saved class ordering if available
    try:
        import json, os
        class_file = os.path.join(os.path.dirname('models'), 'class_names.json')
        # prefer model attribute if set
        class_names = getattr(model, '_class_names', None)
        if class_names is None and os.path.exists('models/class_names.json'):
            class_names = json.load(open('models/class_names.json','r'))
    except Exception:
        class_names = None

    p = float(pred)
    p_diabetic = p
    try:
        if class_names and len(class_names) >= 2:
            # sigmoid corresponds to class_names[1]
            if class_names[1] != 'diabetic':
                p_diabetic = 1.0 - p
    except Exception:
        p_diabetic = p

    prob = p_diabetic * 100.0
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default='models/diabetes_model.h5')
    parser.add_argument('--class-names', type=str, default='models/class_names.json')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit('Model file not found: ' + args.model)

    model = tf.keras.models.load_model(args.model)

    if args.image:
        prob = predict_single(model, args.image)
        print(f'Diabetes probability: {prob:.2f}%')
    elif args.dir:
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            prob = predict_single(model, f)
            print(f, '->', f'{prob:.2f}%')
    else:
        raise SystemExit('Provide --image PATH or --dir PATH')


if __name__ == '__main__':
    main()
