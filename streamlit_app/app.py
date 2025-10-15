import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# --- Configuration ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
# Adjust this path based on where your model will be relative to the Streamlit app
# From streamlit_app/, we go up one level (..) to xray_classifier/, then into models/
MODEL_RELATIVE_PATH = os.path.join('..', 'models', 'cnn_xray_classifier.keras')
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_RELATIVE_PATH))

# Define class names (make sure this matches your dataset's class order from DataLoader)
# This order is usually alphabetical if using flow_from_directory
CLASS_NAMES = ['Normal', 'Condition'] # Example for a binary classifier (e.g., Normal, Pneumonia)

@st.cache_resource # Cache the model loading to prevent reloading on every rerun
def load_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}.")
        st.info("Please ensure the model is trained (`python src/train.py`) and saved correctly.")
        st.stop() # Stop the app if model is not found
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {e}")
        st.info("Check your model file and TensorFlow/Keras installation.")
        st.stop() # Stop the app if model fails to load
    return None

def preprocess_image(img, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Resizes and normalizes the image for model input."""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch dimension
    img_array /= 255.0 # Rescale to [0, 1]
    return img_array

def make_prediction(model, processed_image):
    """Makes a prediction using the loaded model."""
    predictions = model.predict(processed_image)

    if len(CLASS_NAMES) == 2: # Binary classification
        prediction_score = predictions[0][0]
        if prediction_score > 0.5:
            predicted_class = CLASS_NAMES[1]
            confidence = prediction_score
        else:
            predicted_class = CLASS_NAMES[0]
            confidence = 1 - prediction_score
        return predicted_class, confidence
    else: # Multi-class classification
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        return predicted_class, confidence

# --- Streamlit App Interface ---
st.set_page_config(page_title="X-ray Classifier", layout="centered")

st.title("X-ray Image Classifier ")
st.markdown("Upload an X-ray image (e.g., chest X-ray) to classify it.")

# Load the model
model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded X-ray Image', use_column_width=True)
        st.write("") # Add a bit of space

        # Make prediction
        with st.spinner('Analyzing image...'):
            processed_img = preprocess_image(img)
            predicted_class, confidence = make_prediction(model, processed_img)

        st.subheader("Prediction Result:")
        st.success(f"The image is classified as: **{predicted_class}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        if predicted_class == 'Condition': # Customize based on your 'condition' class name
            st.warning("This image is classified as showing a 'Condition'.")
            st.markdown("---")
            st.error("🚨 **Disclaimer:** This is an AI model's prediction and **not a substitute for professional medical advice.** Please consult a qualified healthcare provider for diagnosis and treatment.")
        else:
            st.success("This image is classified as 'Normal'.")
            st.markdown("---")
            st.info("👍 **Disclaimer:** This is an AI model's prediction and **not a substitute for professional medical advice.** Always consult a qualified healthcare provider.")
else:
    st.warning("The classification model could not be loaded. Please check the logs above for errors.")

st.markdown("---")
st.markdown("Built with TensorFlow/Keras and Streamlit.")
st.markdown("[GitHub Repository Link (if applicable)](https://github.com/your-username/xray_classifier)") # Update with your repo link