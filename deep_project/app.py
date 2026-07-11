# ============================================================================
# Streamlit Web App - Cactus Classification
# ============================================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Cactus Classifier",
    page_icon="🌵",
    layout="centered"
)

st.title("🌵 Cactus Image Classifier")
st.write("Upload an image to predict whether it contains a cactus.")

# -------------------------------------------------
# Load Trained Model (EfficientNetV2)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(
        "efficientnetv2_finetuned.keras",
        compile=False,
        custom_objects={
            "preprocess_input": keras.applications.efficientnet_v2.preprocess_input
        }
    )

try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Failed to load model")
    st.code(str(e))   # <-- SHOWS THE REAL ERROR
    st.stop()

# -------------------------------------------------
# Image Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    IMG_SIZE = (224, 224)

    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.efficientnet_v2.preprocess_input(x)

    pred = model.predict(x, verbose=0)[0][0]

    label = "Has Cactus 🌵" if pred >= 0.5 else "No Cactus ❌"
    confidence = pred if pred >= 0.5 else 1 - pred

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")
