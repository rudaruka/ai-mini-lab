
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()
st.title("감정 분석기 💬")
text = st.text_area("문장 입력:")
if st.button("분석하기"):
    if text.strip():
        result = model(text)[0]
        label = "긍정 😊" if result["label"]=="POSITIVE" else "부정 😞"
        st.success(f"결과: {label} (확률 {result['score']:.2f})")
    else:
        st.warning("문장을 입력해야 분석 가능")
