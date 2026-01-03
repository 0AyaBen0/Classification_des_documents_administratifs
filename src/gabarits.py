import cv2
import numpy as np


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def compute_text_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size


def detect_horizontal_lines(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opened, 50, 150)
    return np.sum(edges > 0)


def header_density_ratio(gray):
    h, w = gray.shape
    top = gray[: int(h * 0.25), :]
    rest = gray[int(h * 0.25):, :]

    return (
        compute_text_density(top) /
        (compute_text_density(rest) + 1e-6)
    )


# --------------------------------------------------
# GABARIT SCORES (ROBUST & FUTURE-PROOF)
# --------------------------------------------------
def compute_gabarit_scores(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    aspect_ratio = w / h
    text_density = compute_text_density(gray)
    horizontal_lines = detect_horizontal_lines(gray)
    header_ratio = header_density_ratio(gray)

    scores = {
        "identite": 0.0,
        "releve_bancaire": 0.0,
        "facture_electricite": 0.0,
        "facture_eau": 0.0,
        "document_employeur": 0.0
    }

    # -------------------------------
    # IDENTITY (RECTO + VERSO)
    # -------------------------------
    if aspect_ratio > 1.2 and text_density < 0.07:
        scores["identite"] += 0.6

    # -------------------------------
    # BANK STATEMENT
    # -------------------------------
    if horizontal_lines > 1800 and text_density > 0.15:
        scores["releve_bancaire"] += 0.7

    # -------------------------------
    # INVOICES (STRONG HEADER)
    # -------------------------------
    if header_ratio > 1.5 and 0.10 < text_density < 0.18:
        scores["facture_electricite"] += 0.5
        scores["facture_eau"] += 0.4

    # -------------------------------
    # EMPLOYER DOCUMENT
    # -------------------------------
    if header_ratio < 1.2 and horizontal_lines < 900:
        scores["document_employeur"] += 0.4

    print({
    "aspect_ratio": round(aspect_ratio, 2),
    "text_density": round(text_density, 3),
    "horizontal_lines": horizontal_lines,
    "header_ratio": round(header_ratio, 2)
    })

    return scores


# --------------------------------------------------
# TEST
# --------------------------------------------------
if __name__ == "__main__":
    test_image = "data/preprocessed_images/identite/7/page_1.jpg"
    print("Gabarit scores:", compute_gabarit_scores(test_image))
