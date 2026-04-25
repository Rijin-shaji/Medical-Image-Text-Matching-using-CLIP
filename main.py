from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def predict(image_path):
    model, processor = load_model()

    image = Image.open(image_path).convert("RGB")

    texts = [
        "This is a chest X-ray of a healthy patient",
        "This is a chest X-ray showing signs of pneumonia",
        "This image shows lung infection in a medical scan",
        "This is a normal lung scan with no disease"
    ]

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    print("\n--- Prediction Scores ---")
    for i, text in enumerate(texts):
        print(f"{text}: {probs[0][i].item():.4f}")

    best_idx = probs.argmax().item()
    print("\nBest Match:", texts[best_idx])

if __name__ == "__main__":
    predict("images/sample.jpg")
