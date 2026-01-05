# Architecture du Système

## Vue d'ensemble

Le système de classification de documents administratifs est organisé en 6 modules principaux fonctionnant en mode **100% offline**.

## Architecture Modulaire

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE PRINCIPAL                      │
│                      (main.py)                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│   Module 1   │ │   Module 2  │ │  Module 3  │
│  Offline     │ │  Gabarits   │ │     CV     │
│  Manager     │ │  Detector   │ │  Hybrid    │
└───────┬──────┘ └──────┬───────┘ └─────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│   Module 4   │ │   Module 5  │ │  Module 6  │
│     NLP      │ │   Fusion    │ │ Interface  │
│  Classifier  │ │ Multimodale │ │   Web/CLI  │
└──────────────┘ └─────────────┘ └────────────┘
```

## Module 1: Configuration Offline

**Fichiers:**
- `src/offline_manager.py`
- `setup_offline.py`

**Responsabilités:**
- Téléchargement unique des modèles pré-entraînés
- Gestion du cache des modèles
- Vérification de l'intégrité
- Support de plusieurs versions de modèles

**Modèles gérés:**
- ResNet50, EfficientNet (CV)
- CamemBERT (NLP)
- Tokenizers associés

## Module 2: Système de Gabarits

**Fichiers:**
- `src/gabarits/detector_gabarits.py`
- `models/gabarits/gabarits_config.json`

**Responsabilités:**
- Détection de features structurelles:
  - Ratio d'aspect
  - Détection de photo
  - Structure tabulaire
  - Densité de texte/chiffres
  - Zones de signature
  - Alignement vertical
- Calcul de scores de correspondance par famille

**Technologies:**
- OpenCV (Hough Transform, Contours)
- Haar Cascades (détection visage)

## Module 3: Computer Vision Hybride

**Fichiers:**
- `src/computer_vision/hybrid_cv_model.py`
- `src/computer_vision/train_cv.py`

**Architecture:**
```
Input Image (224x224)
    │
    ├─► CNN Branch (ResNet50/EfficientNet)
    │   └─► Global Average Pooling
    │       └─► Features (2048/1280 dim)
    │
    └─► Gabarits Branch
        └─► Features structurelles (10 dim)
            └─► Dense Layers (64 dim)
                │
                ▼
        ┌───────────────┐
        │   Fusion      │
        │  Concat + FC   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Classifier    │
        │  (5 classes)  │
        └──────────────┘
```

**Fonctionnalités:**
- Fine-tuning du backbone pré-entraîné
- Augmentation de données
- Early stopping
- Learning rate scheduling

## Module 4: NLP Offline

**Fichiers:**
- `src/nlp/ocr_extractor.py`
- `src/nlp/semantic_patterns.py`
- `src/nlp/nlp_classifier.py`

**Pipeline NLP:**
```
Image
  │
  ├─► Preprocessing (denoise, contrast, deskew)
  │
  ├─► OCR (Tesseract)
  │   └─► Text Extraction
  │
  └─► Classification
      ├─► Motifs Sémantiques (mots-clés)
      ├─► CamemBERT (embeddings [CLS])
      └─► Fusion des scores
```

**Approches:**
1. **Motifs Sémantiques**: Détection de mots-clés spécifiques par famille
2. **CamemBERT**: Embeddings pré-entraînés pour le français
3. **RNN** (optionnel): Classification séquentielle avec attention

## Module 5: Fusion Multimodale

**Fichiers:**
- `src/fusion/multimodal_fusion.py`

**Stratégies de Fusion:**

1. **Accord Parfait**
   - CV et NLP donnent la même classe avec confiance > 0.8
   - Confiance = moyenne des deux

2. **CV Fort + Gabarits**
   - CV très confiant (>0.9) + validation gabarits (>0.7)
   - Favorise CV pour documents visuels

3. **NLP Fort + Motifs**
   - NLP très confiant (>0.9) + motifs spécifiques
   - Favorise NLP pour documents textuels

4. **Fusion Pondérée**
   - Combinaison pondérée des scores
   - Validation par règles métier

**Règles Métier:**
- Identité: nécessite photo + ratio d'aspect
- Relevé: nécessite structure tabulaire
- Factures: nécessite unités de mesure
- Employeur: nécessite zone de signature

## Module 6: Pipeline Principal

**Fichiers:**
- `main.py` (CLI)
- `interface.py` (Streamlit Web)

**Workflow:**
```
PDF Input
    │
    ├─► PDF → Images (pdf2image)
    │
    ├─► Pour chaque page:
    │   │
    │   ├─► Preprocessing
    │   │
    │   ├─► Features Gabarits
    │   │
    │   ├─► Classification CV
    │   │
    │   ├─► OCR + Classification NLP
    │   │
    │   └─► Fusion Multimodale
    │       │
    │       └─► Validation Métier
    │
    └─► Tri dans dossiers de sortie
```

## Flux de Données

### Entraînement

```
raw_pdfs/
    │
    ├─► prepare_data.py
    │   └─► images/ (organisées par classe)
    │
    ├─► train.py
    │   └─► models/cv/best_model.pth
```

### Inférence

```
PDF/Image
    │
    ├─► Preprocessing
    ├─► CV Model → Scores CV
    ├─► OCR → Text
    ├─► NLP → Scores NLP
    ├─► Gabarits → Scores Gabarits
    │
    └─► Fusion → Classe Finale
```

## Gestion des Erreurs

- **PDF corrompu**: Skip avec log
- **OCR échoué**: Utiliser seulement CV + Gabarits
- **Modèle non chargé**: Fallback sur motifs sémantiques
- **Confiance faible**: Mettre dans "à_verifier"

## Optimisations

- **Cache des modèles**: Chargement unique en mémoire
- **Batch processing**: Traitement par lots
- **Multithreading**: Pour OCR et classification
- **Modèles légers**: MobileNet optionnel

## Métriques et Monitoring

- **Accuracy par classe**
- **Temps de traitement**
- **Taux de rejet**
- **Logs des décisions difficiles**

## Extensibilité

Le système est conçu pour être facilement extensible:

- **Nouvelles classes**: Ajouter dans `config.yaml` et `gabarits_config.json`
- **Nouveaux modèles**: Implémenter dans les modules correspondants
- **Nouvelles stratégies**: Ajouter dans `MultimodalFusion`
- **Nouvelles règles métier**: Étendre `apply_business_rules()`

