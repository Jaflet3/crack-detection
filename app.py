import streamlit as st
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Concrete Crack Detection System")
st.caption("Image Processing Based Structural Health Monitoring")

# -----------------------------
# FUNCTIONS
# -----------------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blur

def crack_detection(blur):
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blur, 100, 200)
    return thresh, edges

def crack_metrics(thresh, edges):
    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    crack_area = (crack_pixels / total_pixels) * 100
    edge_density = np.sum(edges > 0) / edges.size
    confidence = min(100, (edge_density * 12000))
    return round(crack_area, 2), round(edge_density, 4), round(confidence, 2)

def overlay_crack(original, thresh):
    overlay = original.copy()
    overlay[thresh == 255] = [255, 0, 0]
    return overlay

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Concrete Surface Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray, blur = preprocess_image(image_np)
    thresh, edges = crack_detection(blur)
    crack_area, edge_density, confidence = crack_metrics(thresh, edges)

    # Decision Logic
    if crack_area < 0.2 and edge_density < 0.005:
        result = "No Crack Detected"
        severity = "None"
        recommendation = "Structure is safe"
        show_overlay = False
    else:
        result = "Crack Detected"
        show_overlay = True

        if crack_area < 1.5:
            severity = "Low"
            recommendation = "Monitor periodically"
        elif crack_area < 5:
            severity = "Medium"
            recommendation = "Repair recommended"
        else:
            severity = "High"
            recommendation = "Immediate maintenance required"

    # -----------------------------
    # DISPLAY IMAGES
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.image(image_np, caption="Original Image", use_column_width=True)

    if show_overlay:
        overlay = overlay_crack(image_np, thresh)
        col2.image(overlay, caption="Detected Crack Area", use_column_width=True)
    else:
        col2.image(image_np, caption="No Crack Found", use_column_width=True)

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.divider()
    st.subheader("📊 Analysis Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Crack Area (%)", crack_area)
    c2.metric("Edge Density", edge_density)
    c3.metric("Confidence (%)", confidence)

    if result == "Crack Detected":
        st.error(f"⚠️ Result: {result}")
    else:
        st.success(f"✅ Result: {result}")

    st.info(f"🧱 Severity Level: {severity}")
    st.write(f"🛠 Recommendation: {recommendation}")
