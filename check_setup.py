"""
Script de Vérification de l'Installation
Vérifie que tout est correctement configuré avant d'exécuter le projet
"""

import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Nécessite Python 3.8+")
        return False

def check_tesseract():
    """Vérifie que Tesseract est installé"""
    try:
        result = subprocess.run(
            ['tesseract', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Tesseract installé - {version}")
            
            # Vérifier la langue française
            result_langs = subprocess.run(
                ['tesseract', '--list-langs'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if 'fra' in result_langs.stdout:
                print("✅ Langue française (fra) disponible")
                return True
            else:
                print("⚠️  Langue française (fra) non trouvée")
                return False
        else:
            print("❌ Tesseract non trouvé")
            return False
    except FileNotFoundError:
        print("❌ Tesseract non installé ou non dans le PATH")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification de Tesseract: {e}")
        return False

def check_dependencies():
    """Vérifie que les dépendances Python sont installées"""
    required_packages = [
        'torch', 'torchvision', 'transformers', 'PIL', 'numpy',
        'cv2', 'sklearn', 'pandas', 'pdf2image', 'pytesseract',
        'tensorflow', 'keras', 'streamlit', 'yaml', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('Pillow')
            elif package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✅ {package} installé")
        except ImportError:
            print(f"❌ {package} manquant")
            missing.append(package)
    
    return len(missing) == 0

def check_models():
    """Vérifie que les modèles sont téléchargés"""
    models_dir = Path("models")
    cv_dir = models_dir / "cv"
    nlp_dir = models_dir / "nlp"
    
    cv_ok = False
    nlp_ok = False
    
    if cv_dir.exists():
        cv_files = list(cv_dir.glob("*.pth"))
        if cv_files:
            print(f"✅ Modèles CV trouvés ({len(cv_files)} fichier(s))")
            cv_ok = True
        else:
            print("⚠️  Modèles CV non trouvés - Exécutez: python setup_offline.py")
    else:
        print("⚠️  Dossier models/cv/ non trouvé - Exécutez: python setup_offline.py")
    
    if nlp_dir.exists():
        camembert_dir = nlp_dir / "camembert-base"
        if camembert_dir.exists() and (camembert_dir / "config.json").exists():
            print("✅ Modèle NLP (CamemBERT) trouvé")
            nlp_ok = True
        else:
            print("⚠️  Modèle NLP non trouvé - Exécutez: python setup_offline.py")
    else:
        print("⚠️  Dossier models/nlp/ non trouvé - Exécutez: python setup_offline.py")
    
    return cv_ok and nlp_ok

def check_data():
    """Vérifie que les données sont présentes"""
    data_dir = Path("data/raw_pdfs")
    
    if not data_dir.exists():
        print("⚠️  Dossier data/raw_pdfs/ non trouvé")
        return False
    
    classes = ["identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"]
    total_pdfs = 0
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if class_dir.exists():
            pdfs = list(class_dir.glob("*.pdf"))
            count = len(pdfs)
            total_pdfs += count
            if count > 0:
                print(f"✅ {class_name}: {count} PDF(s)")
            else:
                print(f"⚠️  {class_name}: Aucun PDF trouvé")
        else:
            print(f"⚠️  Dossier {class_name}/ non trouvé")
    
    if total_pdfs > 0:
        print(f"✅ Total: {total_pdfs} PDF(s) trouvé(s)")
        return True
    else:
        print("⚠️  Aucun PDF trouvé dans data/raw_pdfs/")
        return False

def check_structure():
    """Vérifie la structure des dossiers"""
    required_dirs = [
        "src",
        "src/computer_vision",
        "src/nlp",
        "src/gabarits",
        "src/fusion",
        "src/preprocessing",
        "src/utils",
        "data",
        "models"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ existe")
        else:
            print(f"❌ {dir_path}/ manquant")
            all_ok = False
    
    return all_ok

def main():
    """Fonction principale de vérification"""
    print("=" * 60)
    print("VÉRIFICATION DE L'INSTALLATION")
    print("=" * 60)
    print()
    
    results = {
        "Python": check_python_version(),
        "Tesseract": check_tesseract(),
        "Dépendances": check_dependencies(),
        "Structure": check_structure(),
        "Modèles": check_models(),
        "Données": check_data()
    }
    
    print()
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    all_ok = True
    for check, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
        if not result:
            all_ok = False
    
    print()
    if all_ok:
        print("🎉 Tout est prêt ! Vous pouvez commencer à utiliser le système.")
        print()
        print("Prochaines étapes:")
        print("1. python prepare_data.py")
        print("2. python train.py (optionnel)")
        print("3. python main.py --input <fichier> --output <dossier>")
    else:
        print("⚠️  Certaines vérifications ont échoué.")
        print("Consultez GUIDE_EXECUTION.md pour les instructions d'installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()

