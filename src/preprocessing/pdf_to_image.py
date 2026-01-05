"""
Conversion PDF vers images
"""

from pdf2image import convert_from_path
from pathlib import Path
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFToImageConverter:
    """Convertisseur PDF vers images"""
    
    def __init__(self, dpi: int = 150, output_format: str = "PNG"):
        """
        Initialise le convertisseur
        
        Args:
            dpi: Résolution DPI pour la conversion (150 par défaut pour vitesse, 300 pour qualité)
            output_format: Format de sortie ("PNG" ou "JPEG")
        """
        self.dpi = dpi
        self.output_format = output_format
    
    def convert_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Convertit un PDF en images
        
        Args:
            pdf_path: Chemin vers le PDF
            output_dir: Dossier de sortie (optionnel)
            
        Returns:
            Liste des chemins des images créées
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trouvé: {pdf_path}")
        
        try:
            # Convertir le PDF en images
            # Utiliser thread_count pour accélérer si plusieurs pages
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt=self.output_format.lower(),
                thread_count=4  # Paralléliser la conversion (4 threads)
            )
            
            output_paths = []
            
            # Déterminer le dossier de sortie
            if output_dir is None:
                output_dir = pdf_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder chaque page
            for i, image in enumerate(images):
                if len(images) == 1:
                    output_path = output_dir / f"{pdf_path.stem}.{self.output_format.lower()}"
                else:
                    output_path = output_dir / f"{pdf_path.stem}_page_{i+1}.{self.output_format.lower()}"
                
                image.save(output_path, self.output_format)
                output_paths.append(str(output_path))
            
            logger.info(f"PDF {pdf_path.name} converti en {len(images)} image(s)")
            return output_paths
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion de {pdf_path}: {e}")
            raise
    
    def convert_directory(self, pdf_dir: str, output_dir: str, recursive: bool = True):
        """
        Convertit tous les PDFs d'un dossier
        
        Args:
            pdf_dir: Dossier contenant les PDFs
            output_dir: Dossier de sortie
            recursive: Parcourir récursivement les sous-dossiers
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if recursive:
            pdf_files = list(pdf_dir.rglob("*.pdf"))
        else:
            pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Conversion de {len(pdf_files)} PDF(s)...")
        
        for pdf_file in pdf_files:
            try:
                # Créer un sous-dossier pour chaque PDF
                relative_path = pdf_file.relative_to(pdf_dir)
                pdf_output_dir = output_dir / relative_path.parent / pdf_file.stem
                pdf_output_dir.mkdir(parents=True, exist_ok=True)
                
                self.convert_pdf(str(pdf_file), str(pdf_output_dir))
            except Exception as e:
                logger.error(f"Erreur avec {pdf_file}: {e}")
                continue
        
        logger.info("Conversion terminée")


if __name__ == "__main__":
    converter = PDFToImageConverter()
    print("PDFToImageConverter initialisé")

