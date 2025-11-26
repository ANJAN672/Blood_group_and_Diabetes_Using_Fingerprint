import os
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit.components.v1 import html
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer as TFInputLayer
from tensorflow.keras.models import load_model
from keras.src.utils import tf_utils

# Streamlit page config
st.set_page_config(
    page_title="Fingerprint Screening Suite",
    page_icon="🖐️",
    layout="centered",
)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# File paths (resolve relative to this file for robustness)
BASE_DIR = Path(__file__).resolve().parent
BLOOD_MODEL_CANDIDATES = [
    BASE_DIR / "Model" / "keras_model.h5",
    BASE_DIR / "Model" / "blood_keras_model.h5",
    BASE_DIR / "Model" / "db_keras_model.h5",
]
LABEL_CANDIDATES = [
    BASE_DIR / "Model" / "labels.txt",
    BASE_DIR / "Model" / "blood_labels.txt",
    BASE_DIR / "Model" / "db_labels.txt",
]
DIABETES_MODEL_PATH = (
    BASE_DIR
    / "Diabetes-detection-using-fingerprint"
    / "Diabetes_Research-master"
    / "models"
    / "diabetes_model.h5"
)


# Compatibility helpers to support multiple Streamlit versions
def image_responsive(img, caption=None):
    # Prefer new width API; fall back across versions
    try:
        st.image(img, caption=caption, width="stretch")  # modern API
    except TypeError:
        try:
            st.image(img, caption=caption, use_container_width=True)  # mid versions
        except TypeError:
            st.image(img, caption=caption, use_column_width=True)  # older versions


def dataframe_responsive(data):
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        try:
            st.dataframe(data, use_container_width=True)
        except TypeError:
            st.dataframe(data)


def _first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


class _LegacyInputLayer(TFInputLayer):
    """Wrap the TF InputLayer to accept legacy `batch_shape` keys."""

    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super().__init__(*args, **kwargs)


def dtype_policy(config=None, **kwargs):  # pragma: no cover - relies on TF internals
    if isinstance(config, dict) and not kwargs:
        name = config.get("name", "float32")
    else:
        name = kwargs.get("name", config if config is not None else "float32")
    try:
        if hasattr(keras.mixed_precision, "Policy"):
            return keras.mixed_precision.Policy(name)
        if hasattr(keras.mixed_precision, "policy"):
            return keras.mixed_precision.policy.Policy(name)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError(f"Unable to build dtype policy for {name}: {exc}") from exc
    raise ValueError("Mixed precision Policy class unavailable in this TensorFlow build.")


def _compat_keras_tensor(config):
    history = config.get("keras_history", [])
    if isinstance(history, (list, tuple)) and len(history) >= 3:
        return tf_utils.ListWrapper(list(history[:3]))
    raise ValueError(f"Invalid keras tensor config: {config}")


@st.cache_resource
def load_blood_model():
    try:
        blood_model_path = _first_existing(BLOOD_MODEL_CANDIDATES)
        if blood_model_path is None:
            raise FileNotFoundError(
                "No blood group model found. Expected one of: "
                + ", ".join(str(p) for p in BLOOD_MODEL_CANDIDATES)
            )
        return load_model(str(blood_model_path), compile=False)
    except Exception as exc:  # pragma: no cover - surfaced to UI
        raise RuntimeError(f"Failed to load blood group model: {exc}") from exc


@st.cache_data
def load_labels():
    labels_path = _first_existing(LABEL_CANDIDATES)
    if labels_path is None:
        raise RuntimeError(
            "No labels file found. Expected one of: "
            + ", ".join(str(p) for p in LABEL_CANDIDATES)
        )
    with open(labels_path, "r", encoding="utf-8") as f:
        return [label.strip().split(maxsplit=1)[-1] for label in f.readlines()]


@st.cache_resource
def load_diabetes_model():
    if not DIABETES_MODEL_PATH.exists():
        return None

    try:
        return load_model(
            str(DIABETES_MODEL_PATH),
            compile=False,
            custom_objects={
                "InputLayer": _LegacyInputLayer,
                "DTypePolicy": dtype_policy,
                "BatchNormalizationV1": tf.keras.layers.BatchNormalization,
                "KerasTensor": _compat_keras_tensor,
                "__keras_tensor__": _compat_keras_tensor,
            },
        )
    except Exception as exc:  # pragma: no cover - surfaced to UI
        original_convert_inner_node_data = tf_utils.convert_inner_node_data

        def patched_convert_inner_node_data(nested, wrap=False):
            if wrap:
                def _normalize(obj):
                    if isinstance(obj, dict):
                        if (
                            obj.get("class_name") == "__keras_tensor__"
                            and isinstance(obj.get("config", {}).get("keras_history"), (list, tuple))
                        ):
                            history = obj["config"]["keras_history"]
                            return tf_utils.ListWrapper(list(history[:3]))
                        return {k: _normalize(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_normalize(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_normalize(v) for v in obj)
                    return obj

                nested = _normalize(nested)
            return original_convert_inner_node_data(nested, wrap=wrap)

        tf_utils.convert_inner_node_data = patched_convert_inner_node_data  # type: ignore[assignment]

        try:
            return load_model(
                str(DIABETES_MODEL_PATH),
                compile=False,
                custom_objects={
                    "InputLayer": _LegacyInputLayer,
                    "DTypePolicy": dtype_policy,
                    "BatchNormalizationV1": tf.keras.layers.BatchNormalization,
                    "KerasTensor": _compat_keras_tensor,
                    "__keras_tensor__": _compat_keras_tensor,
                },
            )
        except Exception as inner_exc:
            raise RuntimeError(f"Failed to load diabetes model: {inner_exc}") from inner_exc
        finally:
            tf_utils.convert_inner_node_data = original_convert_inner_node_data


# Optional simple auth using Streamlit secrets
# Enable by setting in .streamlit/secrets.toml:
# [auth]
# enabled = true
# users = ["admin"]
# password = "yourpassword"
def check_auth():
    # Only attempt to read Streamlit secrets if a secrets.toml file exists
    home_secrets = Path.home() / ".streamlit" / "secrets.toml"
    proj_secrets = BASE_DIR / ".streamlit" / "secrets.toml"
    secrets_present = home_secrets.exists() or proj_secrets.exists()

    if not secrets_present:
        return True  # Auth disabled when no secrets file

    # Secrets file exists — attempt to read auth config
    try:
        auth_cfg = st.secrets["auth"]  # type: ignore[index]
    except Exception:
        return True  # If auth section is missing or unreadable, disable auth

    if not auth_cfg.get("enabled", False):
        return True

    with st.sidebar:
        st.header("Login")
        user = st.text_input("Username", key="auth_user")
        pwd = st.text_input("Password", type="password", key="auth_pwd")
        btn = st.button("Sign in")
    if btn:
        valid_user = user in set(auth_cfg.get("users", []))
        valid_pwd = pwd == auth_cfg.get("password", "")
        if valid_user and valid_pwd:
            st.session_state["auth_ok"] = True
        else:
            st.error("Invalid credentials")
    return st.session_state.get("auth_ok", False)


def predict_blood_group(image: Image.Image, model=None, labels=None):
    model = model or load_blood_model()
    labels = labels or load_labels()

    size = (224, 224)
    image_data = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_data).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    input_tensor = np.expand_dims(normalized_image_array, axis=0)

    prediction = model.predict(input_tensor, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = labels[predicted_index]
    confidence_score = float(prediction[predicted_index])
    probabilities = {labels[i]: float(round(pred, 4)) for i, pred in enumerate(prediction)}

    return predicted_label, confidence_score, probabilities


def predict_diabetes(image: Image.Image):
    model = load_diabetes_model()
    if model is None:
        return None

    img = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = np.expand_dims(arr, axis=0)

    pred = model.predict(tensor, verbose=0)[0][0]
    class_names = getattr(model, "_class_names", None)
    probability = float(pred)
    try:
        if class_names and len(class_names) >= 2 and class_names[1] != "diabetic":
            probability = 1.0 - probability
    except Exception:
        probability = float(pred)

    return probability


def run_blood_group_tab():
    st.subheader("🩸 Blood Group Detection")
    st.caption("Upload a fingerprint image to predict the likely blood group using a CNN model.")

    uploaded_file = st.file_uploader(
        "📤 Upload fingerprint image",
        type=["jpg", "png", "jpeg", "bmp"],
        key="blood_uploader",
    )
    html(
        """
<script>
const i = setInterval(() => {
  const btn = window.parent.document.querySelector('div[data-testid="stFileUploader"] button');
  if (btn) {
    btn.textContent = 'Upload Image';
    clearInterval(i);
  }
}, 100);
</script>
""",
        height=0,
    )

    try:
        model = load_blood_model()
        labels = load_labels()
    except RuntimeError as err:
        st.error(str(err))
        return

    if not uploaded_file:
        st.info("Upload an image to begin.")
        return

    with st.spinner("Processing image…"):
        time.sleep(1.0)
        image_data = Image.open(uploaded_file).convert("RGB")

    image_responsive(image_data, caption="📷 Uploaded image")

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    for i in range(90):
        time.sleep(0.006)
        progress_bar.progress(i + 1)

    predicted_label, confidence_score, probabilities = predict_blood_group(
        image_data, model=model, labels=labels
    )

    progress_bar.progress(100)
    time.sleep(0.1)
    progress_placeholder.empty()

    st.success(f"Predicted Blood Group: {predicted_label}")
    st.metric(label="Confidence", value=f"{confidence_score:.1%}")

    st.subheader("Class Probabilities")
    dataframe_responsive(
        {
            "Class": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }
    )


def run_diabetes_tab():
    st.subheader("🩺 Diabetes Risk Screening")
    st.caption(
        "Estimate the probability that a fingerprint indicates diabetes. "
        "Ensure the diabetes model is available under `Diabetes-detection-using-fingerprint/Diabetes_Research-master/models/`."
    )

    try:
        diabetes_model = load_diabetes_model()
    except RuntimeError as err:
        st.error(str(err))
        diabetes_model = None

    if diabetes_model is None:
        st.warning(
            "No diabetes model found. Train a model or place `diabetes_model.h5` "
            "in the models directory to enable predictions."
        )
        return

    uploaded = st.file_uploader(
        "📤 Upload fingerprint image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="diabetes_uploader",
    )
    if uploaded is None:
        st.info("Upload an image to get a probability estimate.")
        return

    try:
        img = Image.open(uploaded)
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return

    image_responsive(img, caption="📷 Uploaded image")

    if st.button("Predict diabetes probability", key="predict_diabetes"):
        with st.spinner("Running inference…"):
            probability = predict_diabetes(img)
        if probability is None:
            st.error("Model not available.")
            return

        prob_pct = probability * 100.0
        label = "Likely diabetic" if probability >= 0.5 else "Likely non-diabetic"
        st.metric("Diabetes probability", f"{prob_pct:.2f}%")
        st.write("Assessment:", label)
        st.progress(min(max(int(prob_pct), 0), 100))


def main():
    if not check_auth():
        st.stop()

    st.title("🖐️ Diabetes and Blood Group Detection using Fingerprint")
    st.caption(
        "Unified Streamlit experience for blood group detection and diabetes probability estimation from fingerprint images."
    )

    blood_tab, diabetes_tab = st.tabs(["Blood Group", "Diabetes"])

    with blood_tab:
        run_blood_group_tab()
    with diabetes_tab:
        run_diabetes_tab()

    # Disclaimer
    st.divider()
    st.warning(
        "⚠️ **Disclaimer:** This system cannot be used with fingerprints from babies or infants. "
        "Fingerprint patterns in young children are not fully developed and may lead to inaccurate results. "
        "This tool is intended for use with adult fingerprints only."
    )


if __name__ == "__main__":
    main()
