"""
Script pour extraire uniquement les images de test (15% non utilisées pour l'entraînement)
"""

import sys
from pathlib import Path
import numpy as np
import shutil
import json
from tqdm import tqdm

# Ajouter le dossier src au path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.utils.config_loader import load_config

config = load_config()

# Classes de documents
CLASSES = config.get("classes", [
    "identite",
    "releve_bancaire",
    "facture_electricite",
    "facture_eau",
    "document_employeur"
])

# Paramètres de split (doivent correspondre à train_cv.py)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42  # Même seed que dans train_cv.py pour reproductibilité


def extract_test_images(
    input_dir: str = "data/images",
    output_dir: str = "data/test_images",
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    random_seed: int = RANDOM_SEED
):
    """
    Extrait uniquement les images de test (15% non utilisées pour l'entraînement)
    
    Args:
        input_dir: Dossier contenant toutes les images organisées par classe
        output_dir: Dossier de sortie pour les images de test
        train_split: Proportion pour l'entraînement (0.7)
        val_split: Proportion pour la validation (0.15)
        random_seed: Seed pour reproductibilité (42)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Dossier d'entrée non trouvé: {input_dir}")
    
    # Créer le dossier de sortie
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Statistiques
    total_images = 0
    test_images = 0
    
    test_split = 1.0 - train_split - val_split
    
    print(f"Extraction des images de test depuis {input_dir}...")
    print(f"Split: {train_split*100:.0f}% train, {val_split*100:.0f}% val, {test_split*100:.0f}% test")
    print(f"Seed: {random_seed} (pour reproductibilité)\n")
    
    # Pour chaque classe
    for class_name in CLASSES:
        class_input_dir = input_path / class_name
        class_output_dir = output_path / class_name
        
        if not class_input_dir.exists():
            print(f"⚠️  Dossier {class_input_dir} non trouvé, ignoré")
            continue
        
        # Lister toutes les images
        image_files = list(class_input_dir.glob("*.jpg")) + list(class_input_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"⚠️  Aucune image trouvée dans {class_input_dir}")
            continue
        
        # Mélanger avec le même seed que train_cv.py
        np.random.seed(random_seed)
        image_files_shuffled = image_files.copy()
        np.random.shuffle(image_files_shuffled)
        
        # Calculer les indices de split
        n_total = len(image_files_shuffled)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        # Les images de test sont celles après train + val
        test_files = image_files_shuffled[n_train + n_val:]
        
        total_images += n_total
        test_images += len(test_files)
        
        print(f"📁 {class_name}:")
        print(f"   Total: {n_total} images")
        print(f"   Train: {n_train} images ({(n_train/n_total)*100:.1f}%)")
        print(f"   Val: {n_val} images ({(n_val/n_total)*100:.1f}%)")
        print(f"   Test: {len(test_files)} images ({(len(test_files)/n_total)*100:.1f}%)")
        
        # Créer le dossier de sortie pour cette classe
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier les images de test
        for img_file in tqdm(test_files, desc=f"   Copie {class_name}", leave=False):
            dest_file = class_output_dir / img_file.name
            shutil.copy2(img_file, dest_file)
        
        print()
    
    print(f"\n✅ Extraction terminée!")
    print(f"   Total images: {total_images}")
    print(f"   Images de test extraites: {test_images} ({(test_images/total_images)*100:.1f}%)")
    print(f"   Dossier de sortie: {output_path}")
    
    # Sauvegarder les métadonnées
    metadata = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "random_seed": random_seed,
        "total_images": total_images,
        "test_images": test_images,
        "classes": CLASSES
    }
    
    metadata_path = output_path / "test_set_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   Métadonnées sauvegardées: {metadata_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extrait uniquement les images de test (15% non utilisées pour l'entraînement)"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/images",
        help="Dossier contenant toutes les images organisées par classe"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/test_images",
        help="Dossier de sortie pour les images de test"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=TRAIN_SPLIT,
        help=f"Proportion pour l'entraînement (défaut: {TRAIN_SPLIT})"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help=f"Proportion pour la validation (défaut: {VAL_SPLIT})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Seed pour reproductibilité (défaut: {RANDOM_SEED})"
    )
    
    args = parser.parse_args()
    
    # Calculer test_split
    test_split = 1.0 - args.train_split - args.val_split
    
    if test_split <= 0:
        raise ValueError("train_split + val_split doit être < 1.0")
    
    extract_test_images(
        input_dir=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        random_seed=args.seed
    )
    
    print(f"\n💡 Pour tester uniquement sur ces images de test:")
    print(f"   python main.py --input {args.output} --output results_test/")

