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
MODEL_URL = "https://drive.google.com/uc?id=1PcYOD4I_4dtkYTkKOhAOQMJ6ZTVDn__E"

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

video_url = "https://drive.google.com/uc?export=download&id=1B2fznnDGu9xwYjjspEauyjvkF4umdvBg"
st.video(video_url)
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

    st.image(image_np, channels="BGR", caption="Input Image", use_container_width=True)

    # Run classification
    results = model(image_np)

    # Get predicted class
    class_id = int(results[0].probs.top1)
    class_name = results[0].names[class_id]
    confidence = float(results[0].probs.top1conf)

    # Show prediction
    st.markdown(f"### Predicted Class: **{class_name}**")
    st.markdown(f"ความมั่นใจของโมเดล: **{confidence:.2%}**")
    st.success("Classification complete.")

# Disease-specific information
if class_name.lower() == "melanoma":
    st.markdown("#### 🧠 ข้อมูลเกี่ยวกับ Melanoma")
    st.info("""
    **Melanoma** คือ มะเร็งผิวหนังชนิดรุนแรง เกิดจากเซลล์สร้างเม็ดสี (melanocytes)  
    **ลักษณะอาการ:** ไฝผิดปกติ โตเร็ว สีไม่สม่ำเสมอ ขอบไม่ชัดเจน  
    **สาเหตุ:** โดนรังสี UV มาก, พันธุกรรม, ผิวขาวมาก, เคยมีไฝผิดปกติ  
    """)
elif class_name.lower() == "bcc":
    st.markdown("#### 🧠 ข้อมูลเกี่ยวกับ Basal Cell Carcinoma")
    st.info("""
    **Basal Cell Carcinoma** เป็นมะเร็งผิวหนังชนิดพบได้บ่อยที่สุด  
    **ลักษณะอาการ:** ก้อนใสคล้ายมุก มีเส้นเลือดใต้ผิว ผิวลอกบ่อย ๆ  
    **สาเหตุ:** โดนแดดสะสม, อายุมากขึ้น, ผิวขาว  
    """)
elif class_name.lower() == "scc":
    st.markdown("#### 🧠 ข้อมูลเกี่ยวกับ Squamous Cell Carcinoma")
    st.info("""
    **Squamous Cell Carcinoma** เป็นมะเร็งผิวหนังชนิดแพร่กระจายได้มากกว่าชนิดอื่น  
    **ลักษณะอาการ:** ก้อนแดง ผิวขรุขระ มีเลือดซึม หรือแผลเรื้อรัง  
    **สาเหตุ:** แสง UV, แผลเรื้อรัง, ผิวติดเชื้อ HPV, ภูมิคุ้มกันต่ำ  
    """)
else:
    st.markdown("#### การป้องกันและลดความเสี่ยง")
    st.info("""
    **Squamous Cell Carcinoma** เป็นมะเร็งผิวหนังชนิดแพร่กระจายได้มากกว่าชนิดอื่น  
    **ลักษณะอาการ:** ก้อนแดง ผิวขรุขระ มีเลือดซึม หรือแผลเรื้อรัง  
    **สาเหตุ:** แสง UV, แผลเรื้อรัง, ผิวติดเชื้อ HPV, ภูมิคุ้มกันต่ำ
     """)
