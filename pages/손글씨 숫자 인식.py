import streamlit as st
from PIL import Image
import numpy as np
import pathlib  # 경로 관리를 위한 라이브러리 추가
from tensorflow.keras.models import load_model # Keras load_model 함수 import

st.set_page_config(page_title="손글씨 인식", layout="centered")

st.title("손글씨 숫자 인식 ✏️")

st.markdown(
    """
    <p style='font-size: 1.1em;'>
    0부터 9까지의 손글씨 이미지를 업로드하면, 인공지능 모델이 어떤 숫자인지 예측합니다.
    </p>
    """, 
    unsafe_allow_html=True
)
st.write("---")

# 준비된 사전학습 모델 사용 (가벼운 MNIST)
@st.cache_resource
def load_mnist_model():
    # 1. 현재 파일(pages/손글씨 숫자 인식.py)의 디렉토리를 가져옵니다.
    current_dir = pathlib.Path(__file__).parent 
    
    # 2. current_dir.parent는 상위 폴더(저장소 루트)를 의미합니다.
    #    루트 폴더에 있는 "mnist_model.h5" 파일을 지정합니다.
    model_path = current_dir.parent / "mnist_model.h5" 
    
    # 3. 파일이 실제로 존재하는지 확인하여 디버깅을 돕습니다.
    if not model_path.exists():
        st.error(f"오류: 모델 파일이 예상된 위치에 없습니다! 경로 확인: {model_path}")
        # 파일이 없으면 오류를 발생시켜 재시도를 막습니다.
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # 4. load_model은 문자열 경로를 받으므로, Path 객체를 str()로 변환합니다.
    return load_model(str(model_path)) 

try:
    # 파일을 안전하게 로드합니다.
    model = load_mnist_model()
except FileNotFoundError as e:
    st.stop() # 오류 발생 시 앱 실행을 멈춥니다.


# --- UI 및 예측 로직 ---
img_file = st.file_uploader("여기에 이미지를 드래그하거나 클릭하여 업로드하세요.", type=["png", "jpg", "jpeg"])

if img_file:
    # 1. 업로드된 이미지 표시
    st.image(img_file, caption='업로드된 이미지', use_column_width=True)

    with st.spinner('AI가 이미지를 분석하는 중...'):
        # 2. 이미지 전처리
        # - 흑백(L), 28x28 크기로 조정
        img = Image.open(img_file).convert("L").resize((28, 28))
        # - MNIST 모델에 맞게 색 반전 (흰색 글씨, 검은색 배경)
        arr = 255 - np.array(img)  
        # - 모델 입력 형태 (1, 28, 28, 1)로 변환하고 정규화
        arr = arr.reshape(1, 28, 28, 1) / 255.0

        # 3. 예측 실행
        pred = model.predict(arr)
        predicted_number = np.argmax(pred)
        confidence = pred[0][predicted_number] * 100

        # 4. 결과 출력
        st.success(f"### 예측 숫자: **{predicted_number}**")
        st.info(f"확률: {confidence:.2f}%")

st.write("---")
st.caption("AI Mini Lab - 손글씨 숫자 인식기")
