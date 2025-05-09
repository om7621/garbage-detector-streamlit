import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import pandas as pd
import io

# App title and layout
st.set_page_config(page_title="Garbage Detector", layout="wide")
st.title("♻️ Garbage Type Detector")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.markdown("### Batch Controls")
    if st.button("Clear Batch"):
        st.session_state['batch'] = []

# Initialize batch in session state
if 'batch' not in st.session_state:
    st.session_state['batch'] = []

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

# Function to preprocess image for model
def preprocess(image: Image.Image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

# Inference function
def predict_image(image: Image.Image):
    data = preprocess(image)
    interpreter.set_tensor(input_details['index'], data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details['index'])[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    return labels[idx], confidence

# Main interface: tabs for capture and upload
tab1, tab2 = st.tabs(["Webcam Capture", "Upload Image"])

with tab1:
    img_data = st.camera_input("Capture an image of waste")
    if img_data:
        image = Image.open(img_data).convert("RGB")
        label, score = predict_image(image)
        if score >= threshold:
            st.image(image, caption=f"{label} ({score*100:.2f}%)", use_column_width=True)
            if st.button("Add to Batch", key="webcam_add"):
                st.session_state['batch'].append((image, label, score))
        else:
            st.warning(f"Low confidence: {score*100:.2f}% below threshold")

with tab2:
    uploaded = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        label, score = predict_image(image)
        if score >= threshold:
            st.image(image, caption=f"{label} ({score*100:.2f}%)", use_column_width=True)
            if st.button("Add to Batch", key="upload_add"):
                st.session_state['batch'].append((image, label, score))
        else:
            st.warning(f"Low confidence: {score*100:.2f}% below threshold")

# Display batch results if any
batch = st.session_state['batch']
if batch:
    st.markdown("---")
    st.subheader("Batch Gallery")
    cols = st.columns(3)
    for idx, (img, lbl, scr) in enumerate(batch):
        with cols[idx % 3]:
            st.image(img, caption=f"{lbl} ({scr*100:.2f}%)", use_column_width=True)

    # Create DataFrame for download
    df = pd.DataFrame([{'Label': lbl, 'Confidence': scr} for _, lbl, scr in batch])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Batch Results as CSV",
        data=csv,
        file_name="batch_results.csv",
        mime='text/csv'
    )
