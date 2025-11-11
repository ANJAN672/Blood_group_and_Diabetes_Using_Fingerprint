"""Generate Grad-CAM explanation for a binary Keras model.

Usage example (PowerShell):
F:/diabetes/.venv/Scripts/python.exe .\explain.py --image dataset_labeled\non_diabetic\cluster_0_1001.BMP

Outputs:
 - Prints model probability
 - Saves overlay image next to input path as explain_<basename>.png
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def load_image(path, target_size=(224,224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def find_last_conv_layer(model):
    # heuristic: find last layer with 4D output (batch, h, w, c)
    for layer in reversed(model.layers):
        if hasattr(layer.output, 'shape') and len(layer.output.shape) == 4:
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, eps=1e-8):
    # img_array: (H,W,3) float32 in [0,1]
    img_tensor = tf.expand_dims(img_array, axis=0)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # model output is sigmoid scalar; use that as target
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError('Could not compute gradients (None)')

    # compute channel-wise mean of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    denom = np.max(heatmap) if np.max(heatmap) > 0 else eps
    heatmap /= denom
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, out_path, alpha=0.4, cmap='jet'):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((heatmap.shape[1], heatmap.shape[0]), Image.Resampling.LANCZOS)

    # create heatmap image
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(img)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, extent=(0, img.size[0], img.size[1], 0))
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', default='models/diabetes_model.h5')
    parser.add_argument('--out', default=None, help='Output overlay path')
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f'Image not found: {img_path}')

    if not Path(args.model).exists():
        raise SystemExit(f'Model not found: {args.model}')

    model = tf.keras.models.load_model(args.model)

    # load class ordering if present
    try:
        import json, os
        class_file = os.path.join(os.path.dirname(args.model), 'class_names.json')
        class_names = None
        if os.path.exists(class_file):
            class_names = json.load(open(class_file, 'r'))
        model._class_names = class_names
    except Exception:
        class_names = None

    x = load_image(str(img_path))
    raw = model.predict(np.expand_dims(x, axis=0))[0][0]
    p = float(raw)
    p_diabetic = p
    try:
        if class_names and len(class_names) >= 2 and class_names[1] != 'diabetic':
            p_diabetic = 1.0 - p
    except Exception:
        p_diabetic = p

    print(f'Predicted probability (diabetic): {p_diabetic:.6f}')

    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        print('Could not find a convolutional layer to use for Grad-CAM; aborting explanation.')
        return
    print('Using last conv layer:', last_conv)

    heatmap = make_gradcam_heatmap(x, model, last_conv)

    out_path = args.out
    if out_path is None:
        out_path = img_path.with_name(f'explain_{img_path.stem}.png')

    save_and_display_gradcam(str(img_path), heatmap, str(out_path))
    print('Saved explanation overlay to', out_path)


if __name__ == '__main__':
    main()
