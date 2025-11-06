import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os, pathlib
from PIL import Image
import requests

# --- Configuration ---
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Default relative path (repo structure: /src/streamlit_app/app.py -> ../models/...)
DEFAULT_MODEL_RELATIVE = os.path.join('..', 'models', 'cnn_xray_classifier.keras')
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_RELATIVE))

# Allow override via env vars on Render
MODEL_URL = os.getenv("MODEL_URL")                # e.g., https://.../cnn_xray_classifier.keras
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# Your class names (order must match training)
CLASS_NAMES = ['Normal', 'Condition']

def _download_model(dest_path: str) -> str:
    """Download the model file from MODEL_URL to dest_path (or /tmp fallback)."""
    if not MODEL_URL:
        st.error(f"Model file not found at {dest_path} and MODEL_URL env var is not set.")
        st.stop()

    # Use /tmp if target dir is read-only on the platform
    dest = pathlib.Path(dest_path)
    try_path = dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        test_file = dest.parent / ".write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
    except Exception:
        try_path = pathlib.Path("/tmp/models/cnn_xray_classifier.keras")
        try_path.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Downloading model..."):
        with requests.get(MODEL_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(try_path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    return str(try_path)

@st.cache_resource
def load_model():
    """Ensure the model exists locally and load it once."""
    path = MODEL_PATH
    if not os.path.exists(path):
        path = _download_model(path)

    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        st.stop()

def preprocess_image(img, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    img = img.convert("RGB").resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def make_prediction(model, processed_image):
    preds = model.predict(processed_image)
    # Handle both sigmoid (shape (1,1)) and softmax (shape (1,N))
    preds = np.array(preds)
    if preds.shape[-1] == 1:
        score = float(preds[0][0])
        idx = 1 if score >= 0.5 else 0
        conf = score if idx == 1 else 1 - score
        return CLASS_NAMES[idx], conf
    else:
        idx = int(np.argmax(preds[0]))
        return CLASS_NAMES[idx], float(preds[0][idx])

# ---- Streamlit UI ----
st.set_page_config(page_title="X-ray Classifier", layout="centered")
st.title("X-ray Image Classifier")
st.markdown("Upload an X-ray image (e.g., chest X-ray) to classify it.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray Image', use_column_width=True)
    with st.spinner('Analyzing image...'):
        processed = preprocess_image(img)
        label, conf = make_prediction(model, processed)

    st.subheader("Prediction")
    st.success(f"Class: **{label}**")
    st.info(f"Confidence: **{conf*100:.2f}%**")
    st.markdown("---")
    st.caption("This is not medical advice; consult a medical professional.")
