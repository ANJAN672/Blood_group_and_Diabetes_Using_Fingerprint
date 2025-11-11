from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import os
from typing import Optional

app = FastAPI(title='Fingerprint Diabetes Predictor')

# Global model holder (loaded lazily to avoid import-time dependency on TensorFlow)
_MODEL = None
_MODEL_PATH = os.environ.get('DIABETES_MODEL_PATH', 'models/diabetes_model.h5')


def load_model_if_needed(model_path: Optional[str] = None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = model_path or _MODEL_PATH
    if not os.path.exists(path):
        return None
    # import TensorFlow only when we need to load the model
    import tensorflow as tf
    _MODEL = tf.keras.models.load_model(path)
    return _MODEL


def preprocess_image_bytes(data: bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # model expects batch dimension
    return np.expand_dims(arr, axis=0)


@app.get('/health')
def health():
    model_exists = os.path.exists(_MODEL_PATH)
    return {'status': 'ok', 'model_exists': model_exists, 'model_path': _MODEL_PATH}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # read image bytes
    data = await file.read()
    try:
        x = preprocess_image_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {e}')

    model = load_model_if_needed()
    if model is None:
        raise HTTPException(status_code=500, detail=f'Model not found at {_MODEL_PATH}. Run training (e.g. `python train_diabetes.py --dry-run`) to produce a model.')

    # make prediction
    pred = model.predict(x)[0][0]
    prob = float(pred) * 100.0
    label = 'diabetic' if pred >= 0.5 else 'non_diabetic'

    return JSONResponse({'probability': prob, 'label': label, 'percentage': f'{prob:.2f}%'} )


# Run instructions (comment):
# 1) Ensure you have a trained model at models/diabetes_model.h5 (or set DIABETES_MODEL_PATH env var).
#    To create a quick model: python train_diabetes.py --dry-run --epochs 1
# 2) Install server dependencies: pip install fastapi uvicorn pillow numpy
# 3) Start server: uvicorn predict_api:app --host 0.0.0.0 --port 8000
# 4) POST an image to /predict (multipart/form-data, field name 'file') and receive JSON with probability.
