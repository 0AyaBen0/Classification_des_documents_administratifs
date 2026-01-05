# Guide d'Exécution Étape par Étape

## 📋 Prérequis

Avant de commencer, assurez-vous d'avoir :
- Python 3.8 ou supérieur installé
- Tesseract OCR installé (voir INSTALLATION.md)
- Poppler installé (pour pdf2image)

---

## 🚀 ÉTAPE 1 : Préparation de l'Environnement

### 1.1 Ouvrir un terminal et naviguer vers le projet

```bash
cd Classification_des_documents_administratifs
```

### 1.2 Créer un environnement virtuel (recommandé)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Installer les dépendances Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **Temps estimé : 5-10 minutes**

---

## 🔧 ÉTAPE 2 : Installation de Tesseract OCR

### 2.1 Windows

1. Télécharger depuis : https://github.com/UB-Mannheim/tesseract/wiki
2. Installer avec le package français
3. Vérifier l'installation :
```bash
tesseract --version
tesseract --list-langs
```
(Doit afficher "fra" dans la liste)

### 2.2 Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

### 2.3 macOS

```bash
brew install tesseract tesseract-lang
```

### 2.4 Vérifier l'installation

```bash
tesseract --version
```

---

## 📦 ÉTAPE 3 : Initialisation des Modèles Offline

Cette étape télécharge les modèles pré-entraînés (nécessite internet la première fois).

```bash
python setup_offline.py
```

**Ce que fait cette commande :**
- Télécharge ResNet50 et EfficientNet (modèles CV)
- Télécharge CamemBERT (modèle NLP)
- Sauvegarde les modèles dans `models/`

⏱️ **Temps estimé : 10-20 minutes** (selon la connexion)

**Résultat attendu :**
```
✓ Modèles Computer Vision téléchargés avec succès
✓ Modèles NLP téléchargés avec succès
Initialisation terminée avec succès!
```

---

## 📁 ÉTAPE 4 : Préparation des Données d'Entraînement

Convertir les PDFs en images pour l'entraînement.

```bash
python prepare_data.py --input data/raw_pdfs --output data/images
```

**Ce que fait cette commande :**
- Convertit tous les PDFs de `data/raw_pdfs/` en images PNG
- Organise les images par classe (identite, releve_bancaire, etc.)
- Applique le prétraitement d'images

⏱️ **Temps estimé : 5-15 minutes** (selon le nombre de PDFs)

**Structure créée :**
```
data/images/
├── identite/
├── releve_bancaire/
├── facture_electricite/
├── facture_eau/
└── document_employeur/
```

---

## 🎓 ÉTAPE 5 : Entraînement du Modèle (Optionnel mais Recommandé)

Entraîner le modèle Computer Vision hybride sur vos données.

```bash
python train.py --data-dir data/images --epochs 50 --batch-size 32
```

**Options disponibles :**
- `--epochs` : Nombre d'époques (défaut: 50)
- `--batch-size` : Taille des batches (défaut: 32)
- `--device cuda` : Utiliser GPU si disponible
- `--light` : Utiliser le modèle léger (MobileNet)

**Exemple avec GPU :**
```bash
python train.py --data-dir data/images --epochs 50 --batch-size 64 --device cuda
```

⏱️ **Temps estimé : 30 minutes - 2 heures** (selon le device et la taille des données)

**Résultat :**
- Modèle sauvegardé dans `models/cv/best_model.pth`
- Rapport d'évaluation dans `models/cv/evaluation_report.txt`
- Courbes d'entraînement dans `models/cv/training_history.png`

---

## 🧪 ÉTAPE 6 : Benchmarking (Optionnel)

Comparer les performances des différents composants.

```bash
python benchmark.py --device cpu --runs 10
```

**Ce que fait cette commande :**
- Mesure les temps d'inférence de chaque composant
- Mesure la consommation mémoire
- Génère un rapport JSON

⏱️ **Temps estimé : 2-5 minutes**

**Résultat :**
- Rapport sauvegardé dans `benchmark_results.json`

---

## 🔍 ÉTAPE 7 : Classification de Documents

### 7.1 Classifier un PDF unique

```bash
python main.py --input data/raw_pdfs/identite/1.pdf --output results/
```

**Résultat :**
- Le PDF est classifié
- Les résultats sont affichés dans le terminal (JSON)

### 7.2 Classifier un dossier complet

```bash
python main.py --input data/raw_pdfs/ --output results/
```

**Ce que fait cette commande :**
- Traite tous les PDFs du dossier
- Classe chaque document
- Organise les résultats dans des sous-dossiers

**Structure de sortie :**
```
results/
├── identite/              # Pièces d'identité classées
├── releve_bancaire/      # Relevés bancaires
├── facture_electricite/  # Factures d'électricité
├── facture_eau/          # Factures d'eau
├── document_employeur/   # Documents employeur
├── a_verifier/           # Documents à vérifier manuellement
└── rapport_YYYYMMDD_HHMMSS.json  # Rapport détaillé
```

**Options disponibles :**
- `--device cuda` : Utiliser GPU
- `--light` : Utiliser le modèle léger

**Exemple :**
```bash
python main.py --input test_documents/ --output classified/ --device cuda
```

---

## 🌐 ÉTAPE 8 : Interface Web (Alternative)

Lancer l'interface web Streamlit pour une utilisation interactive.

```bash
streamlit run interface.py
```

**Ce que fait cette commande :**
- Lance un serveur web local
- Ouvre automatiquement votre navigateur
- Interface disponible sur http://localhost:8501

**Fonctionnalités de l'interface :**
- Upload de PDFs via drag & drop
- Visualisation des résultats en temps réel
- Graphiques des scores par méthode
- Téléchargement des résultats en JSON

**Pour arrêter :**
- Appuyer sur `Ctrl+C` dans le terminal

---

## 📊 Exemple de Workflow Complet

Voici un exemple complet du début à la fin :

```bash
# 1. Activer l'environnement virtuel
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/macOS

# 2. Installer les dépendances (première fois seulement)
pip install -r requirements.txt

# 3. Initialiser les modèles (première fois seulement)
python setup_offline.py

# 4. Préparer les données
python prepare_data.py --input data/raw_pdfs --output data/images

# 5. Entraîner le modèle
python train.py --data-dir data/images --epochs 50

# 6. Classifier des documents
python main.py --input data/raw_pdfs/ --output results/

# OU utiliser l'interface web
streamlit run interface.py
```

---

## ⚠️ Dépannage Rapide

### Erreur "TesseractNotFoundError"

**Windows :** Ajouter le chemin dans votre code ou variables d'environnement
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Erreur "Modèle non trouvé"

Vérifier que `setup_offline.py` a bien été exécuté :
```bash
ls models/cv/  # Doit contenir des fichiers .pth
ls models/nlp/  # Doit contenir camembert-base/
```

### Erreur "Pas assez de mémoire"

Utiliser le modèle léger :
```bash
python main.py --input documents/ --output results/ --light
```

### Erreur CUDA

Vérifier que PyTorch avec CUDA est installé :
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Si False, réinstaller PyTorch avec CUDA :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 Notes Importantes

1. **Première exécution** : Les étapes 1-3 doivent être faites une seule fois
2. **Entraînement** : L'étape 5 est optionnelle mais recommandée pour de meilleures performances
3. **Données** : Assurez-vous que vos PDFs sont dans `data/raw_pdfs/` organisés par classe
4. **GPU** : Si vous avez une carte graphique NVIDIA, utilisez `--device cuda` pour accélérer

---

## 🎯 Résumé des Commandes Essentielles

```bash
# Installation (une fois)
pip install -r requirements.txt
python setup_offline.py

# Préparation des données (une fois)
python prepare_data.py

# Entraînement (optionnel)
python train.py

# Classification
python main.py --input <fichier_ou_dossier> --output <dossier_sortie>

# Interface web
streamlit run interface.py
```

---

## ✅ ÉTAPE 0 : Vérification de l'Installation (À FAIRE EN PREMIER)

Avant de commencer, vérifiez que tout est correctement installé :

```bash
python check_setup.py
```

Ce script vérifie :
- ✅ Version de Python
- ✅ Installation de Tesseract OCR
- ✅ Dépendances Python installées
- ✅ Structure des dossiers
- ✅ Modèles téléchargés
- ✅ Données présentes

**Si des erreurs apparaissent, suivez les instructions affichées.**

---

## ✅ Vérification Manuelle (Alternative)

Pour vérifier manuellement que tout fonctionne :

```bash
# Test 1: Vérifier Tesseract
tesseract --version

# Test 2: Vérifier les imports Python
python -c "from src.offline_manager import OfflineModelManager; print('OK')"

# Test 3: Vérifier les modèles
python -c "from src.offline_manager import OfflineModelManager; m = OfflineModelManager(); print('Modèles:', len(m.get_model_info('cv')))"
```

Si tous les tests passent, vous êtes prêt ! 🎉

