import streamlit as st

st.set_page_config(page_title="AI Mini Lab", page_icon="🤖")

st.title("AI bot")
st.write("이 웹사이트는 중학생이 직접 만든 '기초 AI 체험 사이트'입니다!")

st.markdown("""
### 🔍 제공 기능
- 감정 분석기 (텍스트 → 긍정/부정)
- 이미지 분류기 (이미지 → 어떤 물체인지 분류)

왼쪽 메뉴에서 실험을 선택해보세요!
""")
