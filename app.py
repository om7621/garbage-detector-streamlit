import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

st.title("Garbage Type Detector 🌍♻️")

model = load_model("model.h5")

# Define class names as per your Teachable Machine output
class_names = ['Glass', 'Plastic', 'Cardboard', 'Metal', 'Paper', 'Biological', 'Shoes', 'Trash', 'Battery']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # change if your model uses a different size
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Webcam input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"### Prediction: {class_names[class_id]} ({confidence*100:.2f}%)")
