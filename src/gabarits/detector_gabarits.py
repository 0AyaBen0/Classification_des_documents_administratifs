"""
Détecteur de Gabarits
Détecte les features structurelles des documents pour la classification
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectorGabarits:
    """Détecteur de features structurelles basé sur les gabarits"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le détecteur de gabarits
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON des gabarits
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "models" / "gabarits" / "gabarits_config.json"
        
        self.config_path = Path(config_path)
        self.load_config()
        
    def load_config(self):
        """Charge la configuration des gabarits"""
        if not self.config_path.exists():
            logger.warning(f"Fichier de configuration non trouvé: {self.config_path}")
            self.config = {}
            return
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        logger.info(f"Configuration des gabarits chargée: {len(self.config)} familles")
    
    def detect_aspect_ratio(self, image: np.ndarray) -> float:
        """
        Calcule le ratio d'aspect de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Ratio largeur/hauteur
        """
        h, w = image.shape[:2]
        return w / h if h > 0 else 0.0
    
    def detect_photo_zone(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une zone photo (visage)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Tuple (présence, confiance)
        """
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Charger le classificateur de visage Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            # Fallback: détection basée sur les zones rectangulaires avec texture
            # Chercher des zones avec des caractéristiques de photo
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Chercher des rectangles de taille appropriée pour une photo
            h, w = gray.shape
            photo_ratio = 0.15  # Environ 15% de la surface pour une photo d'identité
            
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                total_area = w * h
                
                if 0.05 < area / total_area < 0.3:  # Zone de taille raisonnable
                    aspect = cw / ch if ch > 0 else 0
                    if 0.7 < aspect < 1.3:  # Format carré ou proche
                        return True, 0.6
        
            return False, 0.0
        
        # Détection avec Haar Cascade
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Calculer la confiance basée sur le nombre et la taille des visages
            max_area = max([w * h for (x, y, w, h) in faces])
            total_area = gray.shape[0] * gray.shape[1]
            confidence = min(1.0, max_area / total_area * 10)
            return True, confidence
        
        return False, 0.0
    
    def detect_tabular_structure(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une structure tabulaire
        
        Args:
            image: Image en numpy array
            
        Returns:
            Tuple (présence, score de structure)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détection de lignes horizontales
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)
        
        # Détection de lignes verticales
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)
        
        # Utiliser la transformée de Hough pour détecter les lignes
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Lignes horizontales
        h_lines = cv2.HoughLinesP(
            horizontal_lines, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        # Lignes verticales
        v_lines = cv2.HoughLinesP(
            vertical_lines, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        num_h_lines = len(h_lines) if h_lines is not None else 0
        num_v_lines = len(v_lines) if v_lines is not None else 0
        
        # Score basé sur le nombre de lignes détectées
        # Normaliser par la taille de l'image
        h, w = gray.shape
        total_pixels = h * w
        
        score = min(1.0, (num_h_lines + num_v_lines) / 20.0)
        
        has_structure = (num_h_lines >= 3) and (num_v_lines >= 2)
        
        return has_structure, score
    
    def calculate_text_density(self, image: np.ndarray) -> float:
        """
        Calcule la densité de texte dans l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Densité de texte (0-1)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Appliquer un seuil adaptatif
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Calculer la proportion de pixels de texte
        text_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        density = text_pixels / total_pixels
        return density
    
    def calculate_numeric_density(self, image: np.ndarray) -> float:
        """
        Calcule la densité de chiffres (approximation)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Densité de chiffres (0-1)
        """
        # Approximation: zones avec beaucoup de petits caractères alignés
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détecter les contours de petite taille (caractères)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        total_area = h * w
        
        # Compter les petits contours (probablement des chiffres)
        small_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:  # Taille typique d'un chiffre
                small_contours += area
        
        density = min(1.0, small_contours / total_area * 10)
        return density
    
    def detect_signature_zone(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une zone de signature
        
        Args:
            image: Image en numpy array
            
        Returns:
            Tuple (présence, confiance)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Les signatures sont généralement en bas du document
        bottom_region = gray[int(h * 0.7):, :]
        
        # Détecter les zones avec texture (signature manuscrite)
        # Les signatures ont généralement une texture particulière
        edges = cv2.Canny(bottom_region, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chercher des zones horizontales avec texture
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            
            # Signatures sont généralement larges et peu hautes
            if aspect > 3 and ch < h * 0.1:
                return True, 0.7
        
        return False, 0.0
    
    def detect_vertical_alignment(self, image: np.ndarray) -> float:
        """
        Détecte l'alignement vertical des éléments (pour relevés bancaires)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Score d'alignement (0-1)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détecter les contours verticaux
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=50, maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Analyser l'alignement vertical
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Lignes presque verticales (80-100 degrés)
            if 80 < angle < 100:
                vertical_lines.append((x1 + x2) / 2)
        
        if len(vertical_lines) < 2:
            return 0.0
        
        # Calculer la régularité de l'espacement
        vertical_lines.sort()
        spacings = [vertical_lines[i+1] - vertical_lines[i] for i in range(len(vertical_lines)-1)]
        
        if len(spacings) == 0:
            return 0.0
        
        # Score basé sur la régularité (écart-type faible = alignement régulier)
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings) if len(spacings) > 1 else 0
        
        if mean_spacing == 0:
            return 0.0
        
        regularity = 1.0 / (1.0 + std_spacing / mean_spacing)
        return regularity
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrait toutes les features structurelles d'une image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Dictionnaire de features avec leurs valeurs
        """
        features = {}
        
        # Features de base
        features['aspect_ratio'] = self.detect_aspect_ratio(image)
        
        # Détection photo
        has_photo, photo_conf = self.detect_photo_zone(image)
        features['photo_detection'] = 1.0 if has_photo else 0.0
        features['photo_confidence'] = photo_conf
        
        # Structure tabulaire
        has_table, table_score = self.detect_tabular_structure(image)
        features['tabular_structure'] = 1.0 if has_table else 0.0
        features['tabular_score'] = table_score
        
        # Densités
        features['text_density'] = self.calculate_text_density(image)
        features['numeric_density'] = self.calculate_numeric_density(image)
        
        # Signature
        has_sig, sig_conf = self.detect_signature_zone(image)
        features['signature_zone'] = 1.0 if has_sig else 0.0
        features['signature_confidence'] = sig_conf
        
        # Alignement vertical
        features['vertical_alignment'] = self.detect_vertical_alignment(image)
        
        return features
    
    def calculate_gabarit_scores(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calcule les scores de correspondance avec chaque gabarit
        
        Args:
            image: Image en numpy array
            
        Returns:
            Dictionnaire {nom_gabarit: score}
        """
        features = self.extract_all_features(image)
        scores = {}
        
        for family_name, family_config in self.config.items():
            family_features = family_config.get('features', {})
            total_score = 0.0
            total_weight = 0.0
            
            for feature_name, feature_config in family_features.items():
                if feature_name not in features:
                    continue
                
                weight = feature_config.get('weight', 0.1)
                feature_value = features[feature_name]
                
                # Vérifier les contraintes
                if 'min' in feature_config and feature_value < feature_config['min']:
                    continue
                if 'max' in feature_config and feature_value > feature_config['max']:
                    continue
                
                # Features requises
                if feature_config.get('required', False):
                    if feature_value == 0 or feature_value < 0.5:
                        total_score = 0.0
                        break
                
                # Calculer le score pour cette feature
                if feature_name in ['aspect_ratio', 'text_density', 'numeric_density']:
                    # Features continues: normaliser selon min/max
                    if 'min' in feature_config and 'max' in feature_config:
                        min_val = feature_config['min']
                        max_val = feature_config['max']
                        if min_val < max_val:
                            normalized = (feature_value - min_val) / (max_val - min_val)
                            normalized = max(0, min(1, normalized))
                            score = normalized * weight
                        else:
                            score = weight if min_val <= feature_value <= max_val else 0
                    else:
                        score = feature_value * weight
                else:
                    # Features binaires ou scores
                    score = feature_value * weight
                
                total_score += score
                total_weight += weight
            
            # Normaliser le score
            if total_weight > 0:
                scores[family_name] = total_score / total_weight
            else:
                scores[family_name] = 0.0
        
        return scores
    
    def visualize_features(self, image: np.ndarray, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualise les features détectées sur l'image (pour debugging)
        
        Args:
            image: Image en numpy array
            output_path: Chemin pour sauvegarder l'image (optionnel)
            
        Returns:
            Image annotée
        """
        vis_image = image.copy()
        features = self.extract_all_features(image)
        
        # Dessiner les informations
        h, w = vis_image.shape[:2]
        y_offset = 30
        
        for i, (feature_name, feature_value) in enumerate(features.items()):
            text = f"{feature_name}: {feature_value:.3f}"
            cv2.putText(
                vis_image, text, (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test du détecteur
    detector = DetectorGabarits()
    print("DetectorGabarits initialisé avec succès")
    print(f"Familles de documents configurées: {list(detector.config.keys())}")

