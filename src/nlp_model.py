# src/nlp_model.py

import pytesseract
import cv2
import re
import os

# ------------------------------------------------------------------
# KEYWORDS PER DOCUMENT TYPE (OCR-tolerant)
# ------------------------------------------------------------------
KEYWORDS = {
    "identite": [
        "identite", "identité", "Née", "nationale",
        "nationalité", "carte", "numero", "numéro", 
        "cnie", "etat", "civil"
    ],
    "releve_bancaire": [
        "banque", "compte", "solde", "debit", "débit",
        "credit", "crédit", "operation", "opération", "releve",
        "relevé", "bancaire", "bank", "rib", "iban", "bic",
        "numero", "numéro", "identite", "identité", "reference"
    ],
    "facture_electricite": [
        "electricite", "électricité", "kwh", "kw h",
        "consommation", "abonnement", "puissance"
    ],
    "facture_eau": [
        "eau", "m3", "m³", "consommation", "index", "facture"
    ],
    "document_employeur": [
        "salaire", "employeur", "bulletin", "cotisation",
        "embauche", "attestation", "travail"
    ]
}


# ------------------------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------------
# OCR EXTRACTION
# ------------------------------------------------------------------
def extract_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return ""

    text = pytesseract.image_to_string(
        image,
        lang="fra"
    )
    return clean_text(text)


# ------------------------------------------------------------------
# KEYWORD-BASED CLASSIFICATION
# ------------------------------------------------------------------
def classify_text(text):
    scores = {}

    for doc_class, words in KEYWORDS.items():
        score = 0
        for w in words:
            score += text.count(w)
        scores[doc_class] = score

    max_score = max(scores.values())

    # No keyword detected → fallback
    if max_score == 0:
        return {
            "class": "unknown",
            "confidence": 0.0
        }

    predicted_class = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = max_score / total

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2)
    }


# ------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 🔴 CHANGE THIS PATH TO A REAL IMAGE
    test_image = "data/preprocessed_images/document_employeur/8/page_1.jpg"

    if not os.path.exists(test_image):
        print("[ERROR] Test image not found.")
    else:
        text = extract_text(test_image)

        print("----- OCR TEXT (first 500 chars) -----")
        print(text[:500])
        print("-------------------------------------")

        result = classify_text(text)
        print("Prediction:", result)
