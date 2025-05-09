import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Title
st.title("♻️ Garbage Type Detector (Webcam-based)")

# Load labels
labels = [
    "trash", "shoes", "plastic", "paper", "metal",
    "glass", "clothes", "cardboard", "biological", "battery"
]

# Load model
@st.cache_resource
def load_garbage_model():
    return load_model("model.h5")

model = load_garbage_model()

# Image Preprocessing
def preprocess(image: Image.Image):
    image = image.resize((224, 224))  # Teachable Machine models use 224x224 by default
    img_array = np.asarray(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Webcam input
img_file_buffer = st.camera_input("Capture an image to classify")

if img_file_buffer:
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)

    processed = preprocess(image)
    prediction = model.predict(processed)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    st.success(f"### Predicted: **{labels[predicted_index]}** ({confidence * 100:.2f}%)")
