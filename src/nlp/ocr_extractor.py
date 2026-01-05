"""
Extraction de texte par OCR
"""

import pytesseract
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import re

from ..preprocessing.image_preprocessor import ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extracteur de texte par OCR avec Tesseract"""
    
    def __init__(
        self,
        language: str = "fra",
        psm: int = 6,
        oem: int = 3,
        preprocessing: bool = True
    ):
        """
        Initialise l'extracteur OCR
        
        Args:
            language: Langue pour Tesseract ("fra" pour français)
            psm: Page Segmentation Mode (6 = uniform block of text)
            oem: OCR Engine Mode (3 = default)
            preprocessing: Appliquer le prétraitement d'image
        """
        self.language = language
        self.psm = psm
        self.oem = oem
        self.preprocessing = preprocessing
        
        if preprocessing:
            self.image_preprocessor = ImagePreprocessor()
        
        # Dictionnaire de correction post-OCR
        self.correction_dict = self._load_correction_dict()
    
    def _load_correction_dict(self) -> Dict[str, str]:
        """Charge un dictionnaire de correction pour les erreurs OCR courantes"""
        return {
            # Corrections communes
            "électrîcité": "électricité",
            "électrîcîté": "électricité",
            "consommatîon": "consommation",
            "facturatîon": "facturation",
            "abonnement": "abonnement",
            "abonnemcnt": "abonnement",
            "identîté": "identité",
            "natîonalîté": "nationalité",
            "naîssance": "naissance",
        }
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extrait le texte d'une image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Texte extrait
        """
        # Prétraitement si activé
        if self.preprocessing:
            processed_image = self.image_preprocessor.preprocess_for_ocr(image)
        else:
            processed_image = image
        
        # Configuration Tesseract
        config = f'--oem {self.oem} --psm {self.psm} -l {self.language}'
        
        try:
            # Extraction du texte
            text = pytesseract.image_to_string(processed_image, config=config)
            
            # Correction post-OCR
            text = self._correct_text(text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return ""
    
    def extract_text_with_details(self, image: np.ndarray) -> Dict:
        """
        Extrait le texte avec détails (boîtes, confiance)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Dictionnaire avec texte, boîtes et confiance
        """
        if self.preprocessing:
            processed_image = self.image_preprocessor.preprocess_for_ocr(image)
        else:
            processed_image = image
        
        config = f'--oem {self.oem} --psm {self.psm} -l {self.language}'
        
        try:
            # Extraction avec détails
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=config)
            
            text = pytesseract.image_to_string(processed_image, config=config)
            text = self._correct_text(text)
            
            # Calculer la confiance moyenne
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': text.strip(),
                'boxes': data,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Erreur OCR avec détails: {e}")
            return {
                'text': '',
                'boxes': {},
                'confidence': 0.0
            }
    
    def _correct_text(self, text: str) -> str:
        """
        Corrige les erreurs OCR courantes
        
        Args:
            text: Texte à corriger
            
        Returns:
            Texte corrigé
        """
        corrected = text
        
        # Appliquer les corrections du dictionnaire
        for error, correction in self.correction_dict.items():
            corrected = corrected.replace(error, correction)
        
        # Corrections de patterns communs
        # Espaces multiples
        corrected = re.sub(r'\s+', ' ', corrected)
        
        # Caractères mal reconnus
        char_corrections = {
            '0': 'O',  # Dans certains contextes
            '1': 'I',  # Dans certains contextes
            '5': 'S',  # Dans certains contextes
        }
        
        return corrected
    
    def extract_from_file(self, image_path: str) -> str:
        """
        Extrait le texte d'un fichier image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Texte extrait
        """
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Impossible de charger l'image: {image_path}")
            return ""
        
        return self.extract_text(image)
    
    def extract_from_pdf_pages(self, pdf_path: str) -> List[str]:
        """
        Extrait le texte de toutes les pages d'un PDF
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Liste des textes par page
        """
        from ..preprocessing.pdf_to_image import PDFToImageConverter
        
        converter = PDFToImageConverter()
        image_paths = converter.convert_pdf(pdf_path)
        
        texts = []
        for img_path in image_paths:
            text = self.extract_from_file(img_path)
            texts.append(text)
        
        return texts
    
    def get_text_statistics(self, text: str) -> Dict:
        """
        Calcule des statistiques sur le texte extrait
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire de statistiques
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Compter les chiffres
        numbers = re.findall(r'\d+', text)
        
        # Compter les montants (format monétaire)
        amounts = re.findall(r'\d+[.,]\d{2}', text)
        
        return {
            'total_chars': len(text),
            'total_words': len(words),
            'total_sentences': len([s for s in sentences if s.strip()]),
            'total_numbers': len(numbers),
            'total_amounts': len(amounts),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'numeric_density': len(numbers) / len(text) if text else 0
        }


if __name__ == "__main__":
    extractor = OCRExtractor()
    print("OCRExtractor initialisé")
    print(f"Langue: {extractor.language}")

