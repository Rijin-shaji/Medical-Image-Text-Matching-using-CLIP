import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_model():
    try:
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            force_download=False  
        )
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            force_download=False
        )
        print("Model loaded successfully!\n")
        return model, processor
    except Exception as e:
        print("Error loading model:", e)
        print("\n Try these fixes:")
        print("1. Check internet connection")
        print("2. Run: pip install --upgrade transformers huggingface_hub")
        print("3. Remove any folder named 'openai' in your project")
        exit()


def predict(image_path):
    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        return
    model, processor = load_model()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(" Error loading image:", e)
        return
    texts = [
        "This is a chest X-ray of a healthy patient",
        "This is a chest X-ray showing signs of pneumonia",
        "This image shows lung infection in a medical scan",
        "This is a normal lung scan with no disease"
    ]
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print("\n Prediction Scores \n")
    for i, text in enumerate(texts):
        print(f"{text} → {probs[0][i].item():.4f}")
    best_idx = probs.argmax().item()
    print("\n Best Match:")
    print(texts[best_idx])

if __name__ == "__main__":
    image_path = "D:/New folder (2)/download.jpg"
    predict(image_path)
