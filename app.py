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
MODEL_URL = "https://drive.google.com/uc?id=18bLT5mDOm-I5vVUReN8SldtZS9GJcCWT"
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
    **มะเร็งผิวหนัง** คือ โรคที่เกิดจากเซลล์ผิวหนังเติบโตผิดปกติ ส่วนใหญ่เกิดจากการโดนแสงแดดมากเกินไป โดยเฉพาะแสง UV ซึ่งอาจทำให้เซลล์ผิวเสียหายและกลายเป็นมะเร็งได้.
""", unsafe_allow_html=True)

st.markdown("""**สามารถตรวจมะเร็งผิวหนังชนิด:**

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
if class_name.lower() == "bcc":
    st.markdown("#### 🧠 ข้อมูลเกี่ยวกับ Basal Cell Carcinoma")
    st.info("""
    **Basal Cell Carcinoma** เป็นชนิดที่พบมากที่สุด แต่ไม่ค่อยแพร่กระจาย มักเกิดบริเวณที่ถูกแดดบ่อยๆ เช่น หน้าผาก จมูก หรือหลังคอ ผู้ป่วยส่วนใหญ่ฟื้นตัวได้ดีหากรักษาเร็ว
    
    **ปัจจัยเสี่ยง :** ตากแดดจัดนานๆ ทั้งกลางแจ้งหรือเตียงอาบแดดเป็นประจำ ผิวขาว ผิวไหม้ง่าย ผู้ที่มีอายุมากหรือมีบาดแผลผิวหนังเรื้อรังก็มีความเสี่ยงเพิ่มขึ้น
   
    **สัญญาณเตือน:** มักเป็นก้อนนูนขนาดเล็กผิวมันเงาสีชมพูหรือเนื้อ ตุ่มอาจลักษณะคล้ายมุกหรือตัวตุ่มมักมีขอบนูนตรงกลางยุบ เป็นสะเก็ดหรือมีรอยบุ๋มเล็กๆ บางครั้งเป็นแผลเปื่อยที่ไม่หายง่าย
    """)

else:
    st.markdown("#### วิธีป้องกันมะเร็งผิวหนัง")
    st.info("""
    **หลีกเลี่ยงแดดแรง :** หลีกเลี่ยงการสัมผัสแสงแดดจัดจัดช่วงกลางวัน (ประมาณ 10 โมงเช้าถึงบ่าย 4) และหาเงาร่มรื่นเมื่อต้องอยู่กลางแจ้ง
    
    **ทาครีมกันแดด :** ใช้ครีมกันแดดที่ป้องกันรังสี UVA/UVB ได้ดี (SPF 30 ขึ้นไป) ทุกครั้งก่อนออกแดด และทาซ้ำทุก 2
    
    **สวมใส่อุปกรณ์ป้องกัน :** สวมหมวกปีกกว้าง เสื้อแขนยาวและกางเกงขายาว แว่นตากันแดด เพื่อปกป้องผิวจากแสงแดดโดยตรง
    
    **งดเตียงอาบแดด :** ไม่ใช้เตียงอาบแดด เพราะให้รังสี UV เข้มข้นที่เพิ่มโอกาสเป็นมะเร็งผิวหนัง
    
    **ตรวจผิวหนังเป็นประจำ :** หมั่นสังเกตรอยเปลี่ยนแปลงบนผิวตัวเองเป็นเดือนละ 1 ครั้ง หากพบความผิดปกติควรรีบไปพบแพทย์ผิวหนัง
     """)
