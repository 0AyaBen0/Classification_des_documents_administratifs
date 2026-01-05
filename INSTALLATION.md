# Guide d'Installation

## Prérequis

### Système
- Python 3.8 ou supérieur
- Windows, Linux ou macOS

### Dépendances système

#### Tesseract OCR

**Windows:**
1. Télécharger depuis: https://github.com/UB-Mannheim/tesseract/wiki
2. Installer avec le package de langue français
3. Ajouter au PATH ou configurer `pytesseract.pytesseract.tesseract_cmd`

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

#### Poppler (pour pdf2image)

**Windows:**
- Télécharger depuis: https://github.com/oschwartz10612/poppler-windows/releases
- Extraire et ajouter au PATH

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

## Installation Python

### 1. Cloner le repository

```bash
git clone <repository_url>
cd Classification_des_documents_administratifs
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialiser les modèles offline

```bash
python setup_offline.py
```

Cette étape télécharge les modèles pré-entraînés (ResNet50, EfficientNet, CamemBERT) et peut prendre plusieurs minutes.

### 5. Préparer les données d'entraînement

```bash
python prepare_data.py --input data/raw_pdfs --output data/images
```

### 6. (Optionnel) Entraîner le modèle

```bash
python train.py --data-dir data/images --epochs 50 --batch-size 32
```

## Vérification de l'installation

### Tester Tesseract

```bash
tesseract --version
tesseract --list-langs  # Doit inclure 'fra'
```

### Tester l'import des modules

```python
python -c "from src.offline_manager import OfflineModelManager; print('OK')"
```

## Dépannage

### Erreur Tesseract

Si vous obtenez une erreur `TesseractNotFoundError`:

**Windows:**
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux/macOS:**
Vérifier que Tesseract est dans le PATH:
```bash
which tesseract
```

### Erreur Poppler

Si `pdf2image` ne fonctionne pas:

**Windows:**
Ajouter le chemin de Poppler au PATH ou:
```python
from pdf2image import convert_from_path
# Spécifier le chemin
images = convert_from_path('file.pdf', poppler_path=r'C:\path\to\poppler\bin')
```

### Erreur CUDA

Si vous voulez utiliser GPU mais obtenez des erreurs:

1. Vérifier l'installation de PyTorch avec CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. Si False, réinstaller PyTorch avec CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Modèles non téléchargés

Si les modèles ne se téléchargent pas:

1. Vérifier la connexion internet (première fois uniquement)
2. Vérifier l'espace disque (plusieurs GB nécessaires)
3. Réessayer: `python setup_offline.py`

## Structure après installation

```
Classification_des_documents_administratifs/
├── models/
│   ├── cv/              # Modèles CV téléchargés
│   ├── nlp/             # Modèles NLP téléchargés
│   └── gabarits/        # Configuration gabarits
├── data/
│   ├── raw_pdfs/        # PDFs originaux
│   ├── images/          # Images converties
│   └── output/          # Résultats de classification
└── logs/                 # Fichiers de log
```

## Support

Pour toute question ou problème, consulter:
- Le README.md
- Les logs dans `logs/`
- Les issues du repository

