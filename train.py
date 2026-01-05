"""
Script d'entraînement complet
Entraîne le modèle Computer Vision Hybride
"""

import sys
import argparse
from pathlib import Path

# Ajouter le dossier src au path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.computer_vision.train_cv import train_hybrid_model
from src.utils.config_loader import load_config
from src.utils.logger_setup import setup_logger

config = load_config()
logger = setup_logger()


def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle Computer Vision")
    parser.add_argument(
        "--data-dir", "-d",
        default="data/preprocessed_images",
        help="Dossier contenant les images organisées par classe"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="models/cv",
        help="Dossier de sortie pour le modèle"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Nombre d'époques"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Taille des batches"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.0001,
        help="Taux d'apprentissage"
    )
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Utiliser le modèle léger"
    )
    
    args = parser.parse_args()
    
    # Classes
    classes = config.get("classes", [
        "identite",
        "releve_bancaire",
        "facture_electricite",
        "facture_eau",
        "document_employeur"
    ])
    
    logger.info("Début de l'entraînement...")
    logger.info(f"Classes: {classes}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Modèle léger: {args.light}")
    
    train_hybrid_model(
        data_dir=args.data_dir,
        classes=classes,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        use_light_model=args.light
    )


if __name__ == "__main__":
    main()

