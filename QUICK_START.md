# Guide de Démarrage Rapide

## Installation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Installer Tesseract OCR (voir INSTALLATION.md)

# 3. Initialiser les modèles
python setup_offline.py

# 4. Préparer les données
python prepare_data.py
```

## Utilisation Rapide

### 1. Classifier un PDF unique

```bash
python main.py --input data/raw_pdfs/identite/1.pdf --output results/
```

### 2. Classifier un dossier de PDFs

```bash
python main.py --input data/raw_pdfs/ --output results/
```

### 3. Interface Web

```bash
streamlit run interface.py
```

Ouvrir http://localhost:8501 dans votre navigateur.

## Workflow Complet

### Étape 1: Préparation des données

```bash
# Convertir les PDFs en images
python prepare_data.py --input data/raw_pdfs --output data/images
```

### Étape 2: Entraînement (Optionnel)

```bash
# Entraîner le modèle CV
python train.py --data-dir data/images --epochs 50 --batch-size 32
```

### Étape 3: Classification

```bash
# Classifier des documents
python main.py --input test_documents/ --output classified/
```

### Étape 4: Benchmarking (Optionnel)

```bash
# Comparer les performances des modèles
python benchmark.py --device cpu --runs 10
```

## Structure des Résultats

Après classification, les documents sont organisés dans le dossier de sortie:

```
output/
├── identite/              # Pièces d'identité
├── releve_bancaire/       # Relevés bancaires
├── facture_electricite/    # Factures d'électricité
├── facture_eau/           # Factures d'eau
├── document_employeur/    # Documents employeur
├── a_verifier/            # Documents à vérifier manuellement
└── rapport_YYYYMMDD_HHMMSS.json  # Rapport détaillé
```

## Exemples de Commandes

### Avec GPU

```bash
python main.py --input documents/ --output results/ --device cuda
```

### Avec modèle léger

```bash
python main.py --input documents/ --output results/ --light
```

### Entraînement avec GPU

```bash
python train.py --data-dir data/images --device cuda --batch-size 64
```

## Dépannage Rapide

### Erreur Tesseract

```python
# Dans votre code, ajouter:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Erreur "Modèle non trouvé"

Exécuter: `python setup_offline.py`

### Erreur "Pas de données"

Vérifier que les PDFs sont dans `data/raw_pdfs/` organisés par classe.

## Support

Pour plus de détails, voir:
- `README.md` - Documentation complète
- `INSTALLATION.md` - Guide d'installation détaillé
- `logs/main.log` - Logs du système

