"""
Prétraitement des images
Amélioration de la qualité pour OCR et classification
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Prétraitement des images pour améliorer la qualité"""
    
    def __init__(self):
        """Initialise le préprocesseur"""
        pass
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Réduit le bruit de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image débruitée
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Image vide dans denoise, retour de l'image originale")
                return image
            
            # Vérifier que l'image est valide
            if not isinstance(image, np.ndarray):
                logger.warning("Image n'est pas un numpy array, retour de l'image originale")
                return image
            
            # Vérifier les dimensions
            if len(image.shape) == 3:
                # Image couleur
                if image.shape[2] == 3:
                    try:
                        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                    except cv2.error as e:
                        logger.warning(f"Erreur cv2.fastNlMeansDenoisingColored: {e}, retour de l'image originale")
                        return image
                else:
                    logger.warning(f"Image couleur avec {image.shape[2]} canaux, retour de l'image originale")
                    return image
            elif len(image.shape) == 2:
                # Image en niveaux de gris
                try:
                    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
                except cv2.error as e:
                    logger.warning(f"Erreur cv2.fastNlMeansDenoising: {e}, retour de l'image originale")
                    return image
            else:
                logger.warning(f"Image avec shape invalide {image.shape}, retour de l'image originale")
                return image
        except Exception as e:
            logger.warning(f"Erreur lors du débruitage: {e}, retour de l'image originale")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image avec contraste amélioré
        """
        if len(image.shape) == 3:
            # Convertir en LAB pour améliorer le contraste
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Appliquer CLAHE sur le canal L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Recombiner
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Redresse l'image (correction de l'inclinaison)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image redressée
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Détecter les contours
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculer l'angle moyen
        # cv2.HoughLines retourne un tableau de shape (N, 1, 2)
        # Chaque ligne est un tableau [[rho, theta]]
        angles = []
        try:
            for line in lines[:20]:  # Prendre les 20 premières lignes
                # line est de shape (1, 2), donc line[0] donne [rho, theta]
                if line is not None and len(line) > 0 and len(line[0]) >= 2:
                    rho, theta = line[0][0], line[0][1]
                    angle = (theta * 180 / np.pi) - 90
                    if -45 < angle < 45:
                        angles.append(angle)
        except Exception as e:
            logger.warning(f"Erreur lors du calcul des angles dans deskew: {e}")
            return image
        
        if len(angles) == 0:
            return image
        
        angle = np.median(angles)
        
        # Rotation
        if abs(angle) > 0.5:  # Seuil minimal
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def binarize(self, image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Binarise l'image (noir et blanc)
        
        Args:
            image: Image en numpy array
            method: Méthode ("adaptive", "otsu", "threshold")
            
        Returns:
            Image binarisée
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        else:  # threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary
    
    def resize(self, image: np.ndarray, target_size: Tuple[int, int], keep_aspect: bool = True) -> np.ndarray:
        """
        Redimensionne l'image
        
        Args:
            image: Image en numpy array
            target_size: Taille cible (width, height)
            keep_aspect: Conserver le ratio d'aspect
            
        Returns:
            Image redimensionnée
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        if keep_aspect:
            # Calculer le ratio
            ratio = min(target_w / w, target_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ajouter un padding si nécessaire
            if new_w < target_w or new_h < target_h:
                top = (target_h - new_h) // 2
                bottom = target_h - new_h - top
                left = (target_w - new_w) // 2
                right = target_w - new_w - left
                
                if len(image.shape) == 3:
                    resized = cv2.copyMakeBorder(
                        resized, top, bottom, left, right,
                        cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )
                else:
                    resized = cv2.copyMakeBorder(
                        resized, top, bottom, left, right,
                        cv2.BORDER_CONSTANT, value=255
                    )
            
            return resized
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement optimisé pour OCR
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image prétraitée
        """
        # Débruitage
        image = self.denoise(image)
        
        # Amélioration du contraste
        image = self.enhance_contrast(image)
        
        # Redressement
        image = self.deskew(image)
        
        # Binarisation
        image = self.binarize(image, method="adaptive")
        
        return image
    
    def preprocess_for_classification(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Prétraitement optimisé pour classification (version simplifiée et robuste)
        
        Args:
            image: Image en numpy array
            target_size: Taille cible
            
        Returns:
            Image prétraitée
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Image vide dans preprocess_for_classification")
                return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            # S'assurer que l'image est en uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Redimensionnement d'abord (plus rapide et moins d'erreurs)
            try:
                image = cv2.resize(image, target_size)
            except Exception as e:
                logger.warning(f"Erreur lors du redimensionnement: {e}")
                # Fallback: créer une image noire
                return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            # Amélioration du contraste (optionnel mais utile)
            try:
                image = self.enhance_contrast(image)
            except Exception as e:
                logger.debug(f"Erreur lors de l'amélioration du contraste: {e}, on continue")
            
            # Redressement (optionnel, peut être désactivé si problématique)
            try:
                image = self.deskew(image)
            except Exception as e:
                logger.debug(f"Erreur lors du redressement: {e}, on continue")
            
            # S'assurer que l'image finale est de la bonne taille
            if image.shape[:2] != (target_size[1], target_size[0]):
                image = cv2.resize(image, target_size)
            
            return image
        except Exception as e:
            logger.error(f"Erreur critique dans preprocess_for_classification: {e}")
            # Retourner une image noire de la bonne taille si tout échoue
            try:
                if image is not None and image.size > 0:
                    return cv2.resize(image, target_size)
            except:
                pass
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    def process_image_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        mode: str = "classification"
    ) -> np.ndarray:
        """
        Traite un fichier image
        
        Args:
            input_path: Chemin vers l'image d'entrée
            output_path: Chemin de sortie (optionnel)
            mode: "ocr" ou "classification"
            
        Returns:
            Image prétraitée
        """
        image = cv2.imread(input_path)
        
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {input_path}")
        
        if mode == "ocr":
            processed = self.preprocess_for_ocr(image)
        else:
            processed = self.preprocess_for_classification(image)
        
        if output_path:
            cv2.imwrite(output_path, processed)
            logger.info(f"Image sauvegardée: {output_path}")
        
        return processed


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    print("ImagePreprocessor initialisé")

