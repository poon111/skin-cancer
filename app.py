import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import gdown
import os
from PIL import Image

# Page configuration
st.set_page_config(page_title="Skin Cancer Classification", layout="wide")

# Model download from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1nWNz9YioV_HRWD7ZDsa8YJVDZbhlxJF4"
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }

    section[data-testid="stFileUploader"] label div span {
        color: white !important;
    }

    section[data-testid="stFileUploader"] div span {
        color: black !important;
        font-weight: bold;
    }

    h1, div[data-testid="stMarkdownContainer"], label[data-testid="stFileUploaderLabel"] {
        color: black !important;
        font-family: 'Arial', sans-serif;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Skin Cancer Classification")

# Description
st.markdown("""
มะเร็งผิวหนัง คือ โรคที่เกิดจากเซลล์ผิวหนังเติบโตผิดปกติ ส่วนใหญ่เกิดจากการโดนแสงแดดมากเกินไป โดยเฉพาะแสง UV ซึ่งอาจทำให้เซลล์ผิวเสียหายและกลายเป็นมะเร็งได้.
""", unsafe_allow_html=True)

st.markdown("""สามารถตรวจมะเร็งผิวหนังชนิด:

- Melanoma  
- Basal cell carcinoma  
- Squamous cell carcinoma  
""", unsafe_allow_html=True)

# Camera input
camera_image = st.camera_input("Take a photo using your webcam")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# Determine image source
image = None
if camera_image is not None:
    image = Image.open(camera_image)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# Process image if available
if image is not None:
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image_np, channels="BGR", caption="Input Image", width=500)

    # Run classification
    results = model(image_np)

    # Get predicted class
    class_id = int(results[0].probs.top1)
    class_name = results[0].names[class_id]
    confidence = float(results[0].probs.top1conf)

    # Show prediction
    st.markdown(f"### Predicted Class: **{class_name}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    st.success("Classification complete.")

# Info section
st.markdown("""**สาเหตุและปัจจัยเสี่ยง**

- โดนแสงแดดนานโดยไม่ป้องกัน  
- เคยถูกแดดเผาแรง ๆ โดยเฉพาะในวัยเด็ก  
- ผิวขาว ผมสีอ่อน ตาสีอ่อน  
- มีไฝเยอะ หรือไฝแปลก ๆ  
- คนในครอบครัวเคยเป็นมะเร็งผิวหนัง  

**วิธีป้องกันง่าย ๆ**

- หลีกเลี่ยงแดดแรงช่วง 10 โมงถึงบ่าย 4  
- ใส่เสื้อแขนยาว หมวก และแว่นกันแดด  
- ทาครีมกันแดด SPF 30 ขึ้นไป ทาซ้ำทุก 2 ชั่วโมง  
- หมั่นตรวจผิวตัวเองเดือนละครั้ง  
""", unsafe_allow_html=True)
