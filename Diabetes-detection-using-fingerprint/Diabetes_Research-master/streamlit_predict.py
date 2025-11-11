import os
import streamlit as st
from PIL import Image
import numpy as np
import io

MODEL_PATH = os.environ.get('DIABETES_MODEL_PATH', 'models/diabetes_model.h5')
_MODEL = None


def load_model_if_needed():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    _MODEL = tf.keras.models.load_model(MODEL_PATH)
    # load class name ordering if present so callers can interpret the sigmoid output
    class_json = os.path.join(os.path.dirname(MODEL_PATH), 'class_names.json')
    try:
        import json
        if os.path.exists(class_json):
            _MODEL._class_names = json.load(open(class_json, 'r'))
    except Exception:
        # non-fatal; leave attribute absent if we can't read it
        pass
    return _MODEL


def preprocess_pil(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title('Fingerprint → Diabetes Probability')
    st.write('Upload a fingerprint image and the model will predict the probability (0-100%) that it indicates diabetes.')

    model_status = 'found' if os.path.exists(MODEL_PATH) else 'missing'
    st.sidebar.write(f'Model path: {MODEL_PATH} ({model_status})')

    uploaded = st.file_uploader('Upload fingerprint image', type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'])
    if uploaded is None:
        st.info('Upload an image to get a prediction. If you do not have a trained model, run `python train_diabetes.py --dry-run` to create a test model.')
        return

    try:
        img = Image.open(uploaded)
    except Exception as e:
        st.error(f'Could not read image: {e}')
        return

    st.image(img, caption='Uploaded image', use_container_width=True)

    if st.button('Predict'):
        model = load_model_if_needed()
        if model is None:
            st.error(f'Model not found at {MODEL_PATH}. Train a model or set DIABETES_MODEL_PATH.')
            return

        x = preprocess_pil(img)
        pred = model.predict(x)[0][0]
        # model.predict returns a sigmoid float; depending on training class ordering
        # the value corresponds to the probability of class index 1 (the "second" class
        # in models/class_names.json). Convert to probability for the 'diabetic' class.
        class_names = getattr(model, '_class_names', None)
        # default assumption: sigmoid output is p(diabetic)
        p_diabetic = float(pred)
        try:
            if class_names and len(class_names) >= 2:
                # sigmoid gives prob for class_names[1]
                if class_names[1] != 'diabetic':
                    p_diabetic = 1.0 - float(pred)
        except Exception:
            p_diabetic = float(pred)

        prob = p_diabetic * 100.0
        label = 'Diabetic' if p_diabetic >= 0.5 else 'Non-diabetic'

        st.metric('Prediction', f'{prob:.2f}%')
        st.write('Label:', label)

        # show a simple progress bar reflecting probability
        st.progress(min(max(int(prob), 0), 100))


if __name__ == '__main__':
    main()
