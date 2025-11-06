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
IMG_WIDTH = 150

# Default local path relative to this file: streamlit_app/ -> ../models/...
DEFAULT_MODEL_RELATIVE = os.path.join("..", "models", "cnn_xray_classifier.keras")
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_RELATIVE))

# Allow overrides via env vars (set these in Render if desired)
MODEL_URL = os.getenv("MODEL_URL")  # direct link to the .keras file (S3/Dropbox/GDrive direct/GitHub Release)
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# IMPORTANT: Make sure this order matches the order used during training
CLASS_NAMES = ["Normal", "Condition"]  # adjust if your labels differ


# -----------------------------
# Utilities
# -----------------------------
def _download_model(dest_path: str) -> str:
    """
    Download the model from MODEL_URL to dest_path.
    If the destination directory isn't writable at runtime,
    falls back to /tmp/models/cnn_xray_classifier.keras
    """
    if not MODEL_URL:
        st.error(
            f"Model file not found at {dest_path} and MODEL_URL is not set.\n\n"
            "Set MODEL_URL (a direct-download link) in your Render service's Environment."
        )
        st.stop()

    dest = pathlib.Path(dest_path)

    # Try to write to the intended path; if not possible, use /tmp
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        test_file = dest.parent / ".write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
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
    """
    Ensure the model exists locally (download if missing) and load it once.
    Cached so Streamlit doesn't reload on each interaction.
    """
    path = MODEL_PATH
    if not os.path.exists(path):
        path = _download_model(path)

    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        st.stop()


def preprocess_image(img: Image.Image, target_size=(IMG_WIDTH, IMG_HEIGHT)) -> np.ndarray:
    """
    Convert to RGB, resize, and scale to [0,1] with batch dimension.
    """
    img = img.convert("RGB").resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def make_prediction(model, processed_image: np.ndarray):
    """
    Supports binary (sigmoid) or multi-class (softmax) outputs.
    Returns (label, confidence)
    """
    preds = np.array(model.predict(processed_image))
    if preds.ndim == 2 and preds.shape[1] == 1:
        score = float(preds[0, 0])
        idx = 1 if score >= 0.5 else 0
        conf = score if idx == 1 else (1.0 - score)
        return CLASS_NAMES[idx], conf
    else:
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0, idx])
        return CLASS_NAMES[idx], conf


# -----------------------------
# Streamlit UI
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
        processed = preprocess_image(img)
        label, conf = make_prediction(model, processed)

    st.subheader("Prediction Result")
    st.success(f"Class: **{label}**")
    st.info(f"Confidence: **{conf*100:.2f}%**")

    if label.lower() != "normal":
        st.warning("This suggests an abnormal finding.")
    st.caption("This is an AI aid and not medical advice. Consult a clinician for diagnosis.")

st.markdown("---")
st.caption("Built with TensorFlow/Keras and Streamlit.")
