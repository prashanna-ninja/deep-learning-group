"""
Nepali Sign Language (NSL) Alphabet Recognition - Streamlit demo.

Run after training the notebook:
    pip install streamlit opencv-python tensorflow pillow
    streamlit run app.py

Loads:
    models/mobilenetv2_tl.keras
    models/label_map.json
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Resolve paths relative to this file so `streamlit run app.py` works
# no matter which directory you launch it from.
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_tl.h5"
LABELS_PATH = ROOT / "models" / "label_map.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="NSL Alphabet Recognition", page_icon="\U0001F91F")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # index (int) -> "devanagari (roman)"
    return {int(k): f'{v["devanagari"]} ({v["roman"]})' for k, v in raw.items()}


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32")   # model rescales internally
    return arr[None, ...]


st.title("\U0001F91F Nepali Sign Language - Alphabet Recognition")
st.caption(
    "Upload a hand-sign image or use your webcam. The MobileNetV2 model predicts "
    "the Nepali character. Prototype for CSC60904 - AI for the Himalayas 2026."
)

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        f"Could not find `{MODEL_PATH}` or `{LABELS_PATH}`.\n\n"
        "Train the model in `notebooks/NSL_pipeline.ipynb` first (run through "
        "Section 10). It writes both files into the `models/` folder next to this app."
    )
    st.stop()

model = load_model()
labels = load_labels()

tab_upload, tab_cam = st.tabs(["Upload image", "Use webcam"])

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    img = Image.open(file) if file else None

with tab_cam:
    shot = st.camera_input("Take a photo of the hand sign")
    if shot:
        img = Image.open(shot)

if "img" in dir() and img is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input", use_column_width=True)

    probs = model.predict(preprocess(img), verbose=0)[0]
    top = probs.argsort()[::-1][:3]

    with col2:
        st.subheader("Prediction")
        st.markdown(f"## {labels.get(int(top[0]), top[0])}")
        st.metric("Confidence", f"{probs[top[0]] * 100:.1f}%")
        st.write("Top-3:")
        for i in top:
            st.write(f"- **{labels.get(int(i), i)}** - {probs[i] * 100:.1f}%")
            st.progress(float(probs[i]))

    st.info(
        "Note: this model was trained on posed dataset images. Live webcam accuracy "
        "may drop due to lighting / background / hand-position differences (the "
        "domain gap discussed in the report)."
    )
else:
    st.write("Waiting for an image...")
