import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.title("손글씨 숫자 인식 ✏️")

st.write("0~9 손글씨 이미지를 업로드하면 어떤 숫자인지 예측합니다.")

# 준비된 사전학습 모델 사용 (가벼운 MNIST)
@st.cache_resource
def load_mnist_model():
    return load_model("mnist_model.h5")  # GitHub에 파일 포함 예정

model = load_mnist_model()

img_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if img_file:
    img = Image.open(img_file).convert("L").resize((28, 28))
    arr = 255 - np.array(img)               # 색 반전
    arr = arr.reshape(1, 28, 28, 1) / 255.0

    pred = model.predict(arr)
    st.success(f"예측 숫자: **{np.argmax(pred)}**")
