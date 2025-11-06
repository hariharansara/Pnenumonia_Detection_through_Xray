# streamlit_app/app.py
import os
import pathlib
import requests
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -----------------------------
# Configuration
# -----------------------------
IMG_HEIGHT = 150
IMG_WIDTH  = 150

# Default path relative to this file: streamlit_app/ -> ../models/...
DEFAULT_MODEL_RELATIVE = os.path.join("..", "models", "cnn_xray_classifier.keras")
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_RELATIVE))

# Allow overrides via env vars (set in Render → Environment)
MODEL_URL  = os.getenv("MODEL_URL")  # direct link to .keras
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# MUST match training label order
CLASS_NAMES = ["Normal", "Condition"]


# -----------------------------
# Helpers
# -----------------------------
def _download_model(dest_path: str) -> str:
    """Download model from MODEL_URL to dest_path; fall back to /tmp if not writable."""
    if not MODEL_URL:
        st.error(
            f"Model not found at {dest_path} and MODEL_URL is not set.\n"
            "Add MODEL_URL in Render → Environment (must be a direct-download link)."
        )
        st.stop()

    dest = pathlib.Path(dest_path)

    # Try writing where requested; if RO, switch to /tmp
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        test = dest.parent / ".write_test"
        with open(test, "w") as f: f.write("ok")
        test.unlink(missing_ok=True)
        target = dest
    except Exception:
        target = pathlib.Path("/tmp/models/cnn_xray_classifier.keras")
        target.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Downloading model…"):
        with requests.get(MODEL_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    return str(target)


@st.cache_resource
def load_model():
    """Ensure the model exists locally (download if missing) and load once (cached)."""
    path = MODEL_PATH
    if not os.path.exists(path):
        path = _download_model(path)
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        st.stop()


def preprocess_image(img: Image.Image, target_size=(IMG_WIDTH, IMG_HEIGHT)) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def make_prediction(model, processed_image: np.ndarray):
    preds = np.array(model.predict(processed_image))
    # Binary (sigmoid) vs multi-class (softmax)
    if preds.ndim == 2 and preds.shape[1] == 1:
        score = float(preds[0, 0])
        idx = 1 if score >= 0.5 else 0
        conf = score if idx == 1 else (1.0 - score)
        return CLASS_NAMES[idx], conf
    else:
        idx  = int(np.argmax(preds[0]))
        conf = float(preds[0, idx])
        return CLASS_NAMES[idx], conf


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="X-ray Classifier", layout="centered")
st.title("X-ray Image Classifier")
st.markdown("Upload an X-ray image (e.g., chest X-ray) to classify it.")

model = load_model()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray Image", use_column_width=True)

    with st.spinner("Analyzing image…"):
        proc = preprocess_image(img)
        label, conf = make_prediction(model, proc)

    st.subheader("Prediction")
    st.success(f"Class: **{label}**")
    st.info(f"Confidence: **{conf*100:.2f}%**")
    if label.lower() != "normal":
        st.warning("This suggests an abnormal finding.")
    st.caption("This is an AI aid, not medical advice.")
