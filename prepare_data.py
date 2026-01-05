"""
Script de préparation des données
Convertit les PDFs en images et organise les données pour l'entraînement
"""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm

# Ajouter le dossier src au path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Imports avec chemins absolus
from src.preprocessing.pdf_to_image import PDFToImageConverter
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.utils.logger_setup import setup_logger

logger = setup_logger()


def prepare_training_data(
    raw_pdfs_dir: str = "data/raw_pdfs",
    output_images_dir: str = "data/images",
    preprocess: bool = True,
    dpi: int = 150  # DPI réduit pour accélérer (150 est suffisant pour classification)
):
    """
    Prépare les données d'entraînement en convertissant les PDFs en images
    
    Args:
        raw_pdfs_dir: Dossier contenant les PDFs organisés par classe
        output_images_dir: Dossier de sortie pour les images
        preprocess: Appliquer le prétraitement
    """
    raw_pdfs_path = Path(raw_pdfs_dir)
    output_path = Path(output_images_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Classes
    classes = ["identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"]
    
    # Convertisseur avec DPI réduit pour accélérer
    converter = PDFToImageConverter(dpi=dpi, output_format="PNG")
    
    # Préprocesseur (si nécessaire)
    preprocessor = ImagePreprocessor() if preprocess else None
    
    logger.info("Début de la préparation des données...")
    
    total_converted = 0
    
    for class_name in classes:
        class_pdf_dir = raw_pdfs_path / class_name
        class_img_dir = output_path / class_name
        class_img_dir.mkdir(exist_ok=True)
        
        if not class_pdf_dir.exists():
            logger.warning(f"Dossier {class_pdf_dir} non trouvé")
            continue
        
        # Lister les PDFs
        pdf_files = list(class_pdf_dir.glob("*.pdf"))
        logger.info(f"Traitement de {len(pdf_files)} PDF(s) pour la classe {class_name}")
        
        for pdf_file in tqdm(pdf_files, desc=f"Classe {class_name}"):
            try:
                # Convertir le PDF
                image_paths = converter.convert_pdf(
                    str(pdf_file),
                    str(class_img_dir)
                )
                
                # Préprocesser si demandé
                if preprocess and preprocessor:
                    for img_path in image_paths:
                        preprocessor.process_image_file(
                            img_path,
                            img_path,  # Écraser l'original
                            mode="classification"
                        )
                
                total_converted += len(image_paths)
                
            except Exception as e:
                logger.error(f"Erreur avec {pdf_file}: {e}")
    
    logger.info(f"Conversion terminée: {total_converted} image(s) créée(s)")


def main():
    parser = argparse.ArgumentParser(description="Préparation des données d'entraînement")
    parser.add_argument(
        "--input", "-i",
        default="data/raw_pdfs",
        help="Dossier contenant les PDFs organisés par classe"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/images",
        help="Dossier de sortie pour les images"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Ne pas appliquer le prétraitement"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Résolution DPI pour la conversion (défaut: 150, plus rapide. 300 pour haute qualité)"
    )
    
    args = parser.parse_args()
    
    prepare_training_data(
        raw_pdfs_dir=args.input,
        output_images_dir=args.output,
        preprocess=not args.no_preprocess,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()

