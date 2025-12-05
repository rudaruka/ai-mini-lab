import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return processor, model

processor, model = load_model()

st.title("이미지 분류기 📷")

img_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="업로드된 이미지")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]

    st.success(f"예측 결과: **{label}**")
