# src/cv_model.py

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# RESNET50 FEATURE EXTRACTOR (OFFLINE)
# --------------------------------------------------
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# FEATURE EXTRACTION (FOR FUTURE HYBRID)
# --------------------------------------------------
def extract_visual_features(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(img_tensor)

    return features.squeeze().cpu().numpy()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def compute_text_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size


# --------------------------------------------------
# CV BASELINE CLASSIFICATION (SCORE-BASED)
# --------------------------------------------------
def classify_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {"class": "unknown", "confidence": 0.0}

    h, w, _ = image.shape
    aspect_ratio = w / h

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    text_density = compute_text_density(gray)

    # Initialize scores
    scores = {
        "identite": 0.0,
        "releve_bancaire": 0.0,
        "facture_electricite": 0.0,
        "facture_eau": 0.0,
        "document_employeur": 0.0
    }

    # -------------------------------
    # Identity card
    # -------------------------------
    if len(faces) > 0:
        scores["identite"] += 0.6
    if aspect_ratio > 1.1:
        scores["identite"] += 0.4

    # -------------------------------
    # Bank statement
    # -------------------------------
    scores["releve_bancaire"] += min(text_density * 5, 1.0)

    # -------------------------------
    # Electricity vs Water
    # -------------------------------
    scores["facture_electricite"] += 0.4 + text_density
    scores["facture_eau"] += 0.3 + (0.15 - text_density if text_density < 0.15 else 0)

    # -------------------------------
    # Employer document
    # -------------------------------
    scores["document_employeur"] += 0.2 + (0.1 if aspect_ratio < 1.0 else 0)

    # Normalize & select
    predicted_class = max(scores, key=scores.get)
    confidence = float(
        scores[predicted_class] / (sum(scores.values()) + 1e-6)
    )

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2)
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------
if __name__ == "__main__":
    test_image = "data/preprocessed_images/identite/1/page_1.jpg"

    if not os.path.exists(test_image):
        print("[ERROR] Test image not found.")
    else:
        features = extract_visual_features(test_image)
        result = classify_image(test_image)

        print("Feature vector size:", features.shape)
        print("CV Prediction:", result)
