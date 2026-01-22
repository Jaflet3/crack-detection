import streamlit as st
import numpy as np
import cv2
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
import pyttsx3
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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
# LOAD MODEL
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading CNN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# FUNCTIONS
# -----------------------------
def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB").resize((150, 150))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return float(model.predict(arr)[0][0])

def crack_severity(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    return round((crack_pixels / thresh.size) * 100, 2), thresh

def overlay_crack(img_path, thresh):
    img = cv2.imread(img_path)
    img[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_pdf(result, severity):
    file_name = "Crack_Report.pdf"
    c = canvas.Canvas(file_name, pagesize=A4)
    c.setFont("Helvetica", 14)
    c.drawString(100, 750, "Concrete Crack Detection Report")
    c.drawString(100, 710, f"Result: {result}")
    c.drawString(100, 680, f"Severity: {severity}%")
    c.save()
    return file_name

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Concrete Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    cnn_score = cnn_predict(temp_path)
    severity, thresh = crack_severity(temp_path)

    if cnn_score > 0.6:
        decision = "Crack Detected"
        speak("Warning! Crack detected")
        st.error("⚠️ Crack Detected")
    else:
        decision = "No Crack"
        speak("No crack detected")
        st.success("✅ No Crack Detected")

    col1, col2 = st.columns(2)
    col1.image(img, caption="Original Image", use_column_width=True)
    col2.image(overlay_crack(temp_path, thresh), caption="Crack Visualization", use_column_width=True)

    st.info(f"📊 CNN Confidence: {cnn_score:.2f}")
    st.info(f"📏 Crack Area: {severity}%")

    pdf = generate_pdf(decision, severity)
    with open(pdf, "rb") as f:
        st.download_button("📄 Download Report PDF", f, file_name=pdf)
