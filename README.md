# Projet de Classification de Documents Administratifs - Version Offline

## Description

Système intelligent de classification automatique de documents PDF en 5 catégories :
- Pièce d'identité (CNIE – Recto et Verso)
- Relevé bancaire (Différentes Banques)
- Facture d'électricité (Différentes Régies)
- Facture d'eau (Différentes Régies)
- Document employeur (Bulletins de paie + Attestations de Travail)

## Architecture

Le projet est organisé en 6 modules principaux :

1. **Module 1 : Configuration Offline** - Gestion des modèles pré-entraînés
2. **Module 2 : Système de Gabarits** - Détection de features structurelles
3. **Module 3 : Computer Vision Hybride** - Classification par CNN + Gabarits
4. **Module 4 : NLP Offline** - Classification textuelle (OCR + modèles de langue)
5. **Module 5 : Fusion Multimodale** - Combinaison intelligente CV + NLP
6. **Module 6 : Pipeline Principal** - Interface et traitement par lots

## Installation

### Prérequis

- Python 3.8+
- Tesseract OCR installé sur le système
- Git

### Étapes d'installation

1. Cloner le repository
```bash
git clone <repository_url>
cd Classification_des_documents_administratifs
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

4. Installer Tesseract OCR
- **Windows** : Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
- **Linux** : `sudo apt-get install tesseract-ocr tesseract-ocr-fra`
- **Mac** : `brew install tesseract tesseract-lang`

5. Initialiser les modèles offline
```bash
python setup_offline.py
```

## Utilisation

### Préparation des données

Avant d'entraîner le modèle, convertir les PDFs en images:

```bash
python prepare_data.py --input data/raw_pdfs --output data/images
```

### Entraînement du modèle

```bash
python train.py --data-dir data/images --epochs 50 --batch-size 32
```

### Interface en ligne de commande

Classifier un fichier PDF unique:
```bash
python main.py --input path/to/file.pdf --output path/to/output
```

Classifier un dossier de PDFs:
```bash
python main.py --input path/to/pdfs/ --output path/to/output
```

Options:
- `--device cpu|cuda` : Choisir le device (défaut: cpu)
- `--light` : Utiliser le modèle léger (MobileNet)

### Interface Web (Streamlit)

```bash
streamlit run interface.py
```

L'interface web permet de:
- Uploader un PDF et voir les résultats en temps réel
- Visualiser les scores de chaque méthode (CV, NLP, Gabarits)
- Télécharger les résultats en JSON

## Structure du Projet

```
Classification_des_documents_administratifs/
├── data/
│   ├── raw_pdfs/          # PDFs d'entraînement non traités
│   ├── images/            # Images extraites des PDFs
│   ├── preprocessed_images/  # Images prétraitées
│   └── annotations/       # Annotations et métadonnées
├── models/
│   ├── cv/                # Modèles Computer Vision
│   ├── nlp/               # Modèles NLP
│   └── gabarits/          # Configurations de gabarits
├── src/
│   ├── offline_manager.py # Gestionnaire de modèles offline
│   ├── preprocessing/     # Prétraitement des données
│   ├── gabarits/          # Détection de gabarits
│   ├── computer_vision/   # Classification CV
│   ├── nlp/              # Classification NLP
│   ├── fusion/           # Fusion multimodale
│   └── utils/             # Utilitaires
├── setup_offline.py      # Script d'initialisation
├── main.py              # Pipeline principal
├── interface.py         # Interface Streamlit
├── requirements.txt     # Dépendances
└── README.md           # Documentation
```

## Auteurs

Équipe INDIA-S5 - Pr. CHEFIRA

