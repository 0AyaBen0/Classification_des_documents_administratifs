# Utilisation des Modules - Guide Complet

Ce document explique **où** et **comment** chaque module est utilisé dans le système.

---

## 📊 Vue d'Ensemble du Flux

```
PDF Input
    │
    ├─► PREPROCESSING (pdf_to_image.py, image_preprocessor.py)
    │   └─► Images prétraitées
    │
    ├─► GABARITS (detector_gabarits.py)
    │   └─► Features structurelles + Scores
    │
    ├─► COMPUTER VISION (hybrid_cv_model.py)
    │   └─► Prédiction CV
    │
    ├─► NLP (ocr_extractor.py, semantic_patterns.py, nlp_classifier.py)
    │   └─► Prédiction NLP
    │
    └─► FUSION (multimodal_fusion.py)
        └─► Décision Finale
```

---

## 🔧 Module: PREPROCESSING

### Fichiers:
- `src/preprocessing/pdf_to_image.py`
- `src/preprocessing/image_preprocessor.py`

### Où est-il utilisé?

#### 1. Dans `prepare_data.py` (Préparation des données)
```python
from preprocessing.pdf_to_image import PDFToImageConverter
from preprocessing.image_preprocessor import ImagePreprocessor

# Convertit les PDFs en images
converter = PDFToImageConverter()
image_paths = converter.convert_pdf(pdf_path)

# Prétraite les images
preprocessor = ImagePreprocessor()
preprocessor.process_image_file(img_path, output_path, mode="classification")
```
**Quand:** Lors de la préparation des données d'entraînement

#### 2. Dans `main.py` (Classification)
```python
from preprocessing.pdf_to_image import PDFToImageConverter
from preprocessing.image_preprocessor import ImagePreprocessor

class DocumentClassifier:
    def __init__(self):
        self.pdf_converter = PDFToImageConverter()  # Ligne 63
        self.image_preprocessor = ImagePreprocessor()  # Ligne 64
    
    def classify_pdf(self, pdf_path):
        # Convertit PDF → Images
        image_paths = self.pdf_converter.convert_pdf(pdf_path)  # Ligne 239
    
    def classify_image(self, image_path):
        # Prétraite l'image
        processed_image = self.image_preprocessor.preprocess_for_classification(image)  # Ligne 123
```
**Quand:** À chaque classification de document

#### 3. Dans `src/nlp/ocr_extractor.py` (OCR)
```python
from ..preprocessing.image_preprocessor import ImagePreprocessor

class OCRExtractor:
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()  # Pour prétraiter avant OCR
    
    def extract_text(self, image):
        # Prétraitement optimisé pour OCR
        processed_image = self.image_preprocessor.preprocess_for_ocr(image)
```
**Quand:** Avant l'extraction OCR pour améliorer la qualité

#### 4. Dans `src/computer_vision/train_cv.py` (Entraînement)
```python
from preprocessing.pdf_to_image import PDFToImageConverter
from preprocessing.image_preprocessor import ImagePreprocessor
```
**Quand:** Pour charger et prétraiter les images pendant l'entraînement

---

## 🎯 Module: GABARITS

### Fichiers:
- `src/gabarits/detector_gabarits.py`
- `models/gabarits/gabarits_config.json`

### Où est-il utilisé?

#### 1. Dans `main.py` (Classification)
```python
from gabarits.detector_gabarits import DetectorGabarits

class DocumentClassifier:
    def __init__(self):
        self.gabarit_detector = DetectorGabarits()  # Ligne 67
    
    def classify_image(self, image_path):
        # 1. Extraire toutes les features structurelles
        gabarits_features = self.gabarit_detector.extract_all_features(image)  # Ligne 126
        
        # 2. Calculer les scores de correspondance avec chaque gabarit
        gabarits_scores = self.gabarit_detector.calculate_gabarit_scores(image)  # Ligne 127
        
        # 3. Utiliser les features pour le modèle CV (lignes 148-159)
        gabarit_tensor = torch.tensor([
            gabarits_features.get('aspect_ratio', 0.0),
            gabarits_features.get('photo_detection', 0.0),
            # ... autres features
        ])
        
        # 4. Passer les scores à la fusion (ligne 195)
        result = self.fusion.fuse(cv_pred, nlp_pred, gabarits_scores, ...)
```
**Quand:** Pour chaque image classifiée

#### 2. Dans `src/computer_vision/train_cv.py` (Entraînement)
```python
from gabarits.detector_gabarits import DetectorGabarits

class DocumentDataset(Dataset):
    def __init__(self):
        self.gabarit_detector = DetectorGabarits()  # Pour chaque échantillon
    
    def __getitem__(self, idx):
        # Extraire les features de gabarits pour chaque image
        features = self.gabarit_detector.extract_all_features(img_np)
        
        # Créer le tensor de features (10 features principales)
        gabarit_features = np.array([
            features.get('aspect_ratio', 0.0),
            features.get('photo_detection', 0.0),
            # ... autres features
        ])
        
        return image, gabarit_features, label
```
**Quand:** Pendant la préparation des données d'entraînement

#### 3. Dans `benchmark.py` (Benchmarking)
```python
from gabarits.detector_gabarits import DetectorGabarits

def benchmark_gabarits():
    detector = DetectorGabarits()
    features = detector.extract_all_features(test_image)
```
**Quand:** Pour mesurer les performances du détecteur

---

## 🖼️ Module: COMPUTER VISION

### Fichiers:
- `src/computer_vision/hybrid_cv_model.py` (Modèle)
- `src/computer_vision/train_cv.py` (Entraînement)

### Où est-il utilisé?

#### 1. Dans `train.py` (Script d'entraînement)
```python
from computer_vision.train_cv import train_hybrid_model

# Appelle la fonction d'entraînement
train_hybrid_model(
    data_dir="data/images",
    classes=classes,
    epochs=50,
    batch_size=32
)
```
**Quand:** Pour entraîner le modèle sur vos données

#### 2. Dans `src/computer_vision/train_cv.py` (Fonction d'entraînement)
```python
from .hybrid_cv_model import HybridCVModel, LightHybridCVModel

def train_hybrid_model(...):
    # Créer le modèle
    if use_light_model:
        model = LightHybridCVModel(num_classes=len(classes))
    else:
        model = HybridCVModel(num_classes=len(classes))  # Ligne 187
    
    # Entraîner
    for epoch in range(epochs):
        for images, gabarit_features, labels in train_loader:
            outputs = model(images, gabarit_features)  # Utilise hybrid_cv_model.py
            loss = criterion(outputs, labels)
            loss.backward()
```
**Quand:** Pendant l'entraînement du modèle

#### 3. Dans `main.py` (Classification)
```python
from computer_vision.hybrid_cv_model import HybridCVModel

class DocumentClassifier:
    def __init__(self):
        # Créer le modèle
        self.cv_model = HybridCVModel(num_classes=len(CLASSES))  # Ligne 75
        
        # Charger les poids entraînés
        checkpoint = torch.load("models/cv/best_model.pth")
        self.cv_model.load_state_dict(checkpoint['model_state_dict'])  # Ligne 81
    
    def classify_image(self, image_path):
        # Préparer les données
        img_tensor = transform(image).unsqueeze(0).to(device)
        gabarit_tensor = torch.tensor([...]).unsqueeze(0).to(device)
        
        # Prédiction
        probs, preds = self.cv_model.predict(img_tensor, gabarit_tensor)  # Ligne 163
```
**Quand:** Pour chaque image classifiée

#### 4. Dans `benchmark.py` (Benchmarking)
```python
# Charge et teste le modèle CV
model = manager.load_cv_model("resnet50", device)
output = model(dummy_image, dummy_gabarits)
```
**Quand:** Pour mesurer les performances

---

## 📝 Module: NLP

### Fichiers:
- `src/nlp/ocr_extractor.py` (Extraction OCR)
- `src/nlp/semantic_patterns.py` (Motifs sémantiques)
- `src/nlp/nlp_classifier.py` (Classificateur NLP complet)

### Où est-il utilisé?

#### 1. Dans `main.py` (Classification)
```python
from nlp.nlp_classifier import NLPClassifier

class DocumentClassifier:
    def __init__(self):
        self.nlp_classifier = NLPClassifier(device=device)  # Ligne 93
    
    def classify_image(self, image_path):
        # Classification NLP complète (OCR + motifs + CamemBERT)
        nlp_class, nlp_conf, nlp_scores = self.nlp_classifier.classify_image(image, CLASSES)  # Ligne 173
        
        # Si échec, utiliser seulement les motifs
        text = self.nlp_classifier.extract_text(image)  # Utilise ocr_extractor.py
        pattern_scores = self.nlp_classifier.pattern_matcher.match_patterns(text)  # Utilise semantic_patterns.py
```
**Quand:** Pour chaque image classifiée

#### 2. Dans `src/nlp/nlp_classifier.py` (Intérieur du classificateur)
```python
from .ocr_extractor import OCRExtractor
from .semantic_patterns import SemanticPatternMatcher

class NLPClassifier:
    def __init__(self):
        # OCR
        self.ocr_extractor = OCRExtractor()  # Utilise ocr_extractor.py
        
        # Motifs sémantiques
        self.pattern_matcher = SemanticPatternMatcher()  # Utilise semantic_patterns.py
    
    def classify_image(self, image, classes):
        # 1. Extraction OCR
        text = self.ocr_extractor.extract_text(image)  # Ligne dans ocr_extractor.py
        
        # 2. Classification avec motifs
        pattern_scores = self.pattern_matcher.match_patterns(text)  # Ligne dans semantic_patterns.py
        
        # 3. Classification avec CamemBERT (si disponible)
        camembert_scores = self.camembert.classify(text, classes)
        
        # 4. Fusion des résultats
        final_scores = self._fuse_results(results, classes)
```
**Quand:** Appelé depuis main.py pour chaque image

#### 3. Dans `benchmark.py` (Benchmarking)
```python
from nlp.nlp_classifier import NLPClassifier
from nlp.ocr_extractor import OCRExtractor

# Tester OCR
extractor = OCRExtractor()
text = extractor.extract_text(test_image)
```
**Quand:** Pour mesurer les performances OCR

---

## 🔀 Module: FUSION

### Fichiers:
- `src/fusion/multimodal_fusion.py`

### Où est-il utilisé?

#### 1. Dans `main.py` (Classification finale)
```python
from fusion.multimodal_fusion import MultimodalFusion

class DocumentClassifier:
    def __init__(self):
        self.fusion = MultimodalFusion(
            cv_weight=0.5,
            nlp_weight=0.5,
            gabarits_weight=0.3
        )  # Lignes 97-103
    
    def classify_image(self, image_path):
        # Après avoir obtenu les prédictions CV et NLP
        result = self.fusion.fuse(
            cv_pred,           # Prédiction CV
            nlp_pred,          # Prédiction NLP
            gabarits_scores,   # Scores des gabarits
            pattern_scores,    # Scores des motifs NLP
            gabarits_features  # Features structurelles
        )  # Lignes 192-198
        
        # result = (classe_prédite, confiance, score_rejet, stratégie)
        predicted_class, confidence, rejection_score, strategy = result
```
**Quand:** Pour chaque image, après avoir obtenu toutes les prédictions

#### 2. Dans `src/fusion/multimodal_fusion.py` (Stratégies de fusion)
```python
class MultimodalFusion:
    def fuse(self, cv_pred, nlp_pred, gabarits_scores, ...):
        # Stratégie 1: Accord parfait
        if self.fuse_perfect_agreement(cv_pred, nlp_pred):
            return result
        
        # Stratégie 2: CV fort + gabarits
        if self.fuse_cv_strong(cv_pred, nlp_pred, gabarits_scores):
            return result
        
        # Stratégie 3: NLP fort + motifs
        if self.fuse_nlp_strong(cv_pred, nlp_pred, pattern_scores):
            return result
        
        # Stratégie 4: Fusion pondérée standard
        return self._fuse_weighted(...)
```
**Quand:** Appelé depuis main.py

---

## 📋 Résumé: Ordre d'Exécution

### Lors de la Classification (`main.py`):

1. **PREPROCESSING** (ligne 63-64, 123, 239)
   - Convertit PDF → Images
   - Prétraite les images

2. **GABARITS** (ligne 67, 126-127, 148-159)
   - Extrait les features structurelles
   - Calcule les scores de gabarits
   - Prépare les features pour le modèle CV

3. **COMPUTER VISION** (ligne 75, 163)
   - Charge le modèle (hybrid_cv_model.py)
   - Fait la prédiction avec image + features gabarits

4. **NLP** (ligne 93, 173-184)
   - Extrait le texte (ocr_extractor.py)
   - Trouve les motifs (semantic_patterns.py)
   - Classifie avec NLP (nlp_classifier.py)

5. **FUSION** (ligne 97, 192-198)
   - Combine toutes les prédictions
   - Applique les règles métier
   - Retourne la décision finale

### Lors de l'Entraînement (`train.py` → `train_cv.py`):

1. **PREPROCESSING** (dans DocumentDataset)
   - Charge les images
   - Applique les transformations

2. **GABARITS** (dans DocumentDataset.__getitem__)
   - Extrait les features pour chaque image

3. **COMPUTER VISION** (dans train_hybrid_model)
   - Crée le modèle (hybrid_cv_model.py)
   - Entraîne avec images + features gabarits
   - Sauvegarde le modèle

---

## 🎯 Points Clés

1. **hybrid_cv_model.py** est utilisé:
   - Dans `train_cv.py` pour créer le modèle pendant l'entraînement
   - Dans `main.py` pour charger et utiliser le modèle entraîné

2. **train_cv.py** est utilisé:
   - Directement via `train.py` pour lancer l'entraînement
   - Contient la logique complète d'entraînement

3. **Tous les modules preprocessing** sont utilisés:
   - Dans `prepare_data.py` pour préparer les données
   - Dans `main.py` pour traiter les documents
   - Dans `train_cv.py` pour charger les données d'entraînement

4. **Tous les modules sont interconnectés**:
   - Gabarits → CV (features en entrée)
   - NLP → Fusion (scores en entrée)
   - CV + NLP + Gabarits → Fusion (tous ensemble)

---

## 🔍 Exemple Concret

Quand vous exécutez:
```bash
python main.py --input document.pdf --output results/
```

Voici ce qui se passe:

1. `main.py` ligne 329: Crée `DocumentClassifier()`
2. `main.py` ligne 63-64: Initialise `PDFToImageConverter` et `ImagePreprocessor`
3. `main.py` ligne 67: Initialise `DetectorGabarits`
4. `main.py` ligne 75: Crée `HybridCVModel` (utilise hybrid_cv_model.py)
5. `main.py` ligne 93: Crée `NLPClassifier` (qui utilise ocr_extractor.py et semantic_patterns.py)
6. `main.py` ligne 97: Crée `MultimodalFusion`
7. `main.py` ligne 239: Convertit PDF → Images (utilise pdf_to_image.py)
8. `main.py` ligne 123: Prétraite l'image (utilise image_preprocessor.py)
9. `main.py` ligne 126-127: Extrait features gabarits (utilise detector_gabarits.py)
10. `main.py` ligne 163: Prédiction CV (utilise hybrid_cv_model.py)
11. `main.py` ligne 173: Prédiction NLP (utilise nlp_classifier.py → ocr_extractor.py + semantic_patterns.py)
12. `main.py` ligne 192: Fusion finale (utilise multimodal_fusion.py)

Tous les modules travaillent ensemble ! 🎉

