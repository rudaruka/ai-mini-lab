# /ai-mini-lab/pages/손글씨 숫자 인식.py 파일을 다음과 같이 수정하세요.

import streamlit as st
from PIL import Image
import numpy as np
# tensorflow.keras에서 load_model을 가져옵니다. (코드에 이미 있음)
from tensorflow.keras.models import load_model 

st.title("손글씨 숫자 인식 ✏️")
st.write("0~9 손글씨 이미지를 업로드하면 어떤 숫자인지 예측합니다.")

# 준비된 사전학습 모델 사용 (가벼운 MNIST)
@st.cache_resource
def load_mnist_model():
    # 현재 파일(pages/...)에서 한 단계 상위 폴더(..)를 지정하여 루트의 파일을 로드합니다.
    # **주의: 파일이 반드시 루트 폴더에 있어야 합니다.**
    return load_model("../mnist_model.h5") 

model = load_mnist_model()

img_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if img_file:
    # ... (나머지 코드는 동일)
    img = Image.open(img_file).convert("L").resize((28, 28))
    arr = 255 - np.array(img)
    arr = arr.reshape(1, 28, 28, 1) / 255.0

    pred = model.predict(arr)
    st.success(f"예측 숫자: **{np.argmax(pred)}**")
