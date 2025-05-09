import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

st.title("♻️ Garbage Type Detector (Teachable Machine + Webcam)")

# Load the model
@st.cache_resource
def load_teachable_model():
    return load_model("keras_model.h5", compile=False)

model = load_teachable_model()

# Load labels
class_names = open("labels.txt", "r").readlines()
class_names = [label.strip().split(" ", 1)[-1] for label in class_names]

# Image input from webcam
img_file_buffer = st.camera_input("Capture a garbage image")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Output
    st.success(f"### Prediction: {class_name} ({confidence_score * 100:.2f}%)")
