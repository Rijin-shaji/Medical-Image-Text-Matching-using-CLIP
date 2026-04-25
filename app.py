import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

st.title(" Medical Image–Text Matching (CLIP)")
st.write("Upload a medical image and get the best matching description")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

texts = [
    "This is a chest X-ray of a healthy patient",
    "This is a chest X-ray showing signs of pneumonia",
    "This image shows lung infection in a medical scan",
    "This is a normal lung scan with no disease"
]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]

    st.subheader("Prediction Scores:")
    for i, text in enumerate(texts):
        st.write(f"{text}: {probs[i].item():.4f}")

    best_idx = probs.argmax().item()
    st.success(f"Best Match: {texts[best_idx]}")
