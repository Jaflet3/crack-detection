import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import tflite_runtime.interpreter as tflite

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Concrete Crack Detection System")
st.caption("CNN-based Structural Health Monitoring")

# -----------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------
MODEL_PATH = "crack_model.tflite"
FILE_ID = "your_google_drive_file_id"

def download_model(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    import requests
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading CNN model..."):
        download_model(FILE_ID, MODEL_PATH)

# -----------------------------
# LOAD TFLITE MODEL
# -----------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Concrete Surface Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((150, 150))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.divider()
    st.subheader("📊 Result")

    if prediction >= 0.5:
        st.error(f"⚠️ Crack Detected\n\nConfidence: {prediction*100:.2f}%")
        st.info("🛠 Recommendation: Inspection and repair advised")
    else:
        st.success(f"✅ No Crack Detected\n\nConfidence: {(1-prediction)*100:.2f}%")
        st.info("🧱 Structure appears safe")
