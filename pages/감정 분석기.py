import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    # requirements.txt에 torch가 있다면 PyTorch 사용을 명시적으로 강제
    return pipeline("sentiment-analysis", framework="pt")

model = load_model()

st.title("감정 분석기 💬")

text = st.text_area("문장을 입력하세요:")

if st.button("분석하기"):
    if text.strip():
        result = model(text)[0]
        label = "긍정 😊" if result["label"] == "POSITIVE" else "부정 😞"
        st.success(f"결과: {label} (확률 {result['score']:.2f})")
    else:
        st.warning("문장을 입력해야 분석할 수 있어요!")
