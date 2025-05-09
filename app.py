import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

st.title("♻️ Garbage Detector (TFLite)")

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# Load labels
labels = [line.strip().split(" ", 1)[-1] for line in open("labels.txt")]

# Webcam input
img_data = st.camera_input("Capture an image of waste")

if img_data:
    image = Image.open(img_data).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    arr = (np.asarray(image).astype(np.float32) / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)

    # Run inference
    interpreter.set_tensor(input_details["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details["index"])[0]

    # Display result
    idx = int(np.argmax(preds))
    confidence = preds[idx]
    st.success(f"**Prediction:** {labels[idx]} ({confidence * 100:.2f}%)")
