"""
Script d'initialisation offline
Télécharge une seule fois tous les modèles nécessaires pour le fonctionnement offline
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

from src.offline_manager import OfflineModelManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Fonction principale d'initialisation"""
    logger.info("=" * 60)
    logger.info("Initialisation du système offline")
    logger.info("=" * 60)
    
    # Initialiser le gestionnaire
    manager = OfflineModelManager()
    
    # Télécharger les modèles Computer Vision
    logger.info("\n[1/2] Téléchargement des modèles Computer Vision...")
    try:
        manager.download_cv_models()
        logger.info("✓ Modèles Computer Vision téléchargés avec succès")
    except Exception as e:
        logger.error(f"✗ Erreur lors du téléchargement des modèles CV: {e}")
        return False
    
    # Télécharger les modèles NLP
    logger.info("\n[2/2] Téléchargement des modèles NLP...")
    try:
        manager.download_nlp_models()
        logger.info("✓ Modèles NLP téléchargés avec succès")
    except Exception as e:
        logger.error(f"✗ Erreur lors du téléchargement des modèles NLP: {e}")
        return False
    
    # Vérifier l'intégrité des modèles
    logger.info("\nVérification de l'intégrité des modèles...")
    cv_info = manager.get_model_info("cv")
    nlp_info = manager.get_model_info("nlp")
    
    logger.info(f"Modèles CV téléchargés: {len(cv_info)}")
    logger.info(f"Modèles NLP téléchargés: {len(nlp_info)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Initialisation terminée avec succès!")
    logger.info("=" * 60)
    logger.info("\nVous pouvez maintenant utiliser le système en mode offline.")
    logger.info("Pour vérifier Tesseract OCR, exécutez: tesseract --version")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

