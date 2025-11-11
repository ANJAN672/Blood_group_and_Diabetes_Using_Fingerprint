"""Inspect model outputs across dataset_labeled and print stats.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = 'models/diabetes_model.h5'
CLASS_JSON = 'models/class_names.json'
DATA_DIR = Path('dataset_labeled')

if not Path(MODEL_PATH).exists():
    raise SystemExit('Model not found: ' + MODEL_PATH)
if not Path(CLASS_JSON).exists():
    print('Warning: class_names.json not found, proceeding')
else:
    with open(CLASS_JSON,'r') as f:
        class_names = json.load(f)
    print('class_names.json:', class_names)

model = tf.keras.models.load_model(MODEL_PATH)
print('\nModel summary (last 5 layers):')
for l in model.layers[-5:]:
    print(' -', l.name, getattr(l, 'activation', None))

# helper
def load_image(p, target=(224,224)):
    img = Image.open(p).convert('RGB')
    img = img.resize(target, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype='float32')/255.0
    return arr

# gather files
files = []
for cls in sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()]):
    for f in (DATA_DIR/cls).iterdir():
        if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
            files.append((cls, str(f)))

if not files:
    raise SystemExit('No images found in dataset_labeled')

print('\nFound %d images across %d classes' % (len(files), len(set(c for c,_ in files))))

probs = []
per_class = {}
for cls, path in files:
    x = load_image(path)
    p = float(model.predict(x[None,...])[0][0])
    probs.append((cls,path,p))
    per_class.setdefault(cls, []).append(p)

print('\nOverall probability stats:')
allp = np.array([p for _,_,p in probs])
print(' count', len(allp), 'mean', allp.mean(), 'min', allp.min(), 'max', allp.max())

print('\nPer-class stats:')
for cls, arr in per_class.items():
    a = np.array(arr)
    print(f' {cls}: n={len(a)}, mean={a.mean():.4f}, min={a.min():.4f}, max={a.max():.4f}, pct>=0.5={(a>=0.5).mean():.2f}')

print('\nFirst 10 predictions:')
for cls,path,p in probs[:10]:
    print(cls, f'{p:.6f}', Path(path).name)

# check if all predictions are >=0.5
all_ge = (allp>=0.5).all()
print('\nAll predictions >=0.5?', all_ge)

# show class ordering used when creating dataset_labeled
print('\nDataset folders seen (sorted):', sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()]))
