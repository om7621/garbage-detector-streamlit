import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import base64
from datetime import datetime

# Page config & background image
st.set_page_config(page_title="Garbage Detector", layout="wide")
# Sidebar background upload
with st.sidebar.expander("Background Settings"):
    bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])
    if bg_file:
        data = bg_file.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f"<style>.stApp {{background-image: url(data:image/png;base64,{b64}); background-size: cover;}}</style>", unsafe_allow_html=True)

# App title
st.title("♻️ Garbage Type Detector")

# Initialize session state for batch results and feedback
if "results" not in st.session_state:
    st.session_state.results = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.header("Inputs")
    mode = st.radio("Input Mode", ("Webcam", "Upload Image"))

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()
# Load labels
labels = [line.strip().split(" ", 1)[-1] for line in open("labels.txt")]

# Function to process an image and record result
def classify_and_record(image: Image.Image):
    img = ImageOps.fit(image.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(img).astype(np.float32) / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details["index"])[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    # record
    st.session_state.results.append({
        "timestamp": datetime.now(),
        "predicted": labels[idx],
        "confidence": confidence
    })
    return img, labels[idx], confidence

# Input handling
if mode == "Webcam":
    img_data = st.camera_input("Capture an image of waste")
else:
    img_data = st.file_uploader("Upload an image of waste", type=["png", "jpg", "jpeg"])

if img_data:
    image = Image.open(img_data)
    st.image(image, caption="Input Image", use_column_width=True)
    img, pred, conf = classify_and_record(image)
    if conf >= threshold:
        st.success(f"**Prediction:** {pred} ({conf*100:.2f}%)")
    else:
        st.warning(f"Low confidence ({conf*100:.2f}%), predicted: {pred}")
    # Feedback option
    with st.expander("Is this wrong? Provide correct label"):
        correct = st.selectbox("Correct Label", labels, key=len(st.session_state.feedback))
        if st.button("Submit Feedback", key=f"fb_{len(st.session_state.feedback)}"):
            st.session_state.feedback.append({
                "timestamp": datetime.now(),
                "predicted": pred,
                "correct": correct,
                "confidence": conf
            })
            st.success("Feedback recorded!")

# Display batch gallery
if st.session_state.results:
    st.markdown("---")
    st.header("Session Results")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)
    # Download CSV
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Results CSV", data=csv, file_name="results.csv")

# Display feedback log
if st.session_state.feedback:
    st.markdown("---")
    st.header("Feedback Log")
    fb_df = pd.DataFrame(st.session_state.feedback)
    st.dataframe(fb_df)
    fb_csv = fb_df.to_csv(index=False).encode()
    st.download_button("Download Feedback CSV", data=fb_csv, file_name="feedback.csv")
