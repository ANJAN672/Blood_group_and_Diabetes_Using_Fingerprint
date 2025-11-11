import os
import numpy as np

# Suppress TF excessive logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

MODEL_PATH = r"f:\diabetes\Diabetes_Research-master\models\diabetes_model.h5"

print("Loading model:", MODEL_PATH, flush=True)
model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel summary:", flush=True)
model.summary()

# Find last layer with weights
layers_with_weights = [l for l in model.layers if l.weights]
if not layers_with_weights:
    print("No layers with weights found.")
    raise SystemExit(0)

last = layers_with_weights[-1]
weights = last.get_weights()
print("\nLast trainable layer:")
print("  name:", last.name)
print("  class:", last.__class__.__name__)
print("  num_weight_arrays:", len(weights))
for i, w in enumerate(weights):
    print(f"  WEIGHT_{i}_SHAPE:", w.shape)

# Try to interpret as (W, b)
W = None
b = None
if len(weights) == 2:
    W, b = weights
elif len(weights) == 1:
    W = weights[0]
else:
    # Heuristic: last is bias
    W = weights[0]
    b = weights[-1]

if b is not None:
    print("\nBias values (b):")
    # Usually size 1 for binary sigmoid output
    print(b.tolist())
else:
    print("\nNo explicit bias vector found for the last layer.")

if W is not None:
    flat = W.flatten()
    size = flat.size
    print("\nWeights (W) summary:")
    print("  total_params:", int(size))
    print("  min:", float(np.min(flat)))
    print("  max:", float(np.max(flat)))
    print("  mean:", float(np.mean(flat)))
    print("  std:", float(np.std(flat)))

    # Print a small preview and save full values if large
    preview_n = 20
    print(f"  preview_first_{preview_n}:", flat[:preview_n].tolist())

    if size <= 2000:
        print("\nAll weights (flattened):")
        print(flat.tolist())
    else:
        out_csv = r"f:\diabetes\last_layer_weights.csv"
        np.savetxt(out_csv, flat, delimiter=",")
        print("\nFull flattened weights saved to:", out_csv)

print("\nNote: The model outputs p = sigmoid(z) where z = W·f + b.")
print("To reproduce a specific p (e.g., 0.447), you need the feature vector f for that exact image (the CNN's penultimate-layer activations).", flush=True)