"""
Pipeline Principal
Script principal pour la classification de documents
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

# Ajouter le dossier src au path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Imports avec chemins absolus depuis src
from src.utils.config_loader import load_config
from src.utils.logger_setup import setup_logger
from src.preprocessing.pdf_to_image import PDFToImageConverter
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.gabarits.detector_gabarits import DetectorGabarits
from src.computer_vision.hybrid_cv_model import HybridCVModel
from src.nlp.nlp_classifier import NLPClassifier
from src.fusion.multimodal_fusion import MultimodalFusion
from src.offline_manager import OfflineModelManager

# Configuration
config = load_config()
logger = setup_logger(log_file="logs/main.log")

# Classes de documents
CLASSES = config.get("classes", [
    "identite",
    "releve_bancaire",
    "facture_electricite",
    "facture_eau",
    "document_employeur"
])


class DocumentClassifier:
    """Classificateur de documents complet"""
    
    def __init__(self, device: str = "cpu", use_light_model: bool = False):
        """
        Initialise le classificateur
        
        Args:
            device: Device (cpu ou cuda)
            use_light_model: Utiliser le modèle léger
        """
        self.device = device
        self.use_light_model = use_light_model
        
        logger.info("Initialisation des composants...")
        
        # Gestionnaire de modèles offline
        self.model_manager = OfflineModelManager()
        
        # Préprocesseurs (DPI réduit pour accélérer)
        self.pdf_converter = PDFToImageConverter(dpi=150)  # 150 DPI suffisant pour classification
        self.image_preprocessor = ImagePreprocessor()
        
        # Détecteur de gabarits
        self.gabarit_detector = DetectorGabarits()
        
        # Modèle CV
        try:
            if use_light_model:
                from src.computer_vision.hybrid_cv_model import LightHybridCVModel
                self.cv_model = LightHybridCVModel(num_classes=len(CLASSES), gabarit_features_dim=10)
            else:
                self.cv_model = HybridCVModel(num_classes=len(CLASSES), gabarit_features_dim=10)
            
            # Charger les poids si disponibles
            model_path = Path("models/cv/best_model.pth")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=device)
                self.cv_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Modèle CV chargé depuis best_model.pth")
            else:
                logger.warning("Modèle CV non entraîné. Utilisation du modèle pré-entraîné uniquement.")
            
            self.cv_model = self.cv_model.to(device)
            self.cv_model.eval()
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle CV: {e}")
            self.cv_model = None
        
        # Classificateur NLP
        self.nlp_classifier = NLPClassifier(device=device)
        
        # Fusion multimodale
        fusion_config = config.get("fusion", {})
        self.fusion = MultimodalFusion(
            cv_weight=fusion_config.get("cv_weight", 0.5),
            nlp_weight=fusion_config.get("nlp_weight", 0.5),
            gabarits_weight=fusion_config.get("gabarits_weight", 0.3),
            confidence_threshold=fusion_config.get("confidence_threshold", 0.8),
            rejection_threshold=fusion_config.get("rejection_threshold", 0.5)
        )
        
        logger.info("Classificateur initialisé")
    
    def classify_image(self, image_path: str) -> dict:
        """
        Classifie une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Dictionnaire avec les résultats
        """
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Prétraitement
        processed_image = self.image_preprocessor.preprocess_for_classification(image)
        
        # 1. Features de gabarits
        gabarits_features = self.gabarit_detector.extract_all_features(image)
        gabarits_scores = self.gabarit_detector.calculate_gabarit_scores(image)
        
        # 2. Classification CV
        cv_pred = None
        cv_conf = 0.0
        if self.cv_model:
            try:
                # Préparer l'image pour le modèle
                from torchvision import transforms
                from PIL import Image
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil_image).unsqueeze(0).to(self.device)
                
                # Features gabarits pour le modèle
                gabarit_tensor = torch.tensor([
                    gabarits_features.get('aspect_ratio', 0.0),
                    gabarits_features.get('photo_detection', 0.0),
                    gabarits_features.get('photo_confidence', 0.0),
                    gabarits_features.get('tabular_structure', 0.0),
                    gabarits_features.get('tabular_score', 0.0),
                    gabarits_features.get('text_density', 0.0),
                    gabarits_features.get('numeric_density', 0.0),
                    gabarits_features.get('signature_zone', 0.0),
                    gabarits_features.get('signature_confidence', 0.0),
                    gabarits_features.get('vertical_alignment', 0.0)
                ], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Prédiction
                with torch.no_grad():
                    probs, preds = self.cv_model.predict(img_tensor, gabarit_tensor)
                
                cv_class_idx = preds[0].item()
                cv_class = CLASSES[cv_class_idx]
                cv_conf = probs[0][cv_class_idx].item()
                cv_pred = (cv_class, cv_conf)
            except Exception as e:
                logger.error(f"Erreur classification CV: {e}")
        
        # 3. Classification NLP
        nlp_class, nlp_conf, nlp_scores = self.nlp_classifier.classify_image(image, CLASSES)
        nlp_pred = (nlp_class, nlp_conf) if nlp_class else None
        
        if nlp_pred is None:
            # Fallback: utiliser les motifs sémantiques uniquement
            text = self.nlp_classifier.extract_text(image)
            if text:
                pattern_scores = self.nlp_classifier.pattern_matcher.match_patterns(text)
                if pattern_scores:
                    nlp_class = max(pattern_scores, key=pattern_scores.get)
                    nlp_conf = pattern_scores[nlp_class]
                    nlp_pred = (nlp_class, nlp_conf)
        
        if nlp_pred is None:
            nlp_pred = (CLASSES[0], 0.0)  # Fallback
        
        # 4. Fusion multimodale
        if cv_pred:
            pattern_scores = nlp_scores.get('patterns', {}) if isinstance(nlp_scores, dict) else {}
            result = self.fusion.fuse(
                cv_pred,
                nlp_pred,
                gabarits_scores,
                pattern_scores,
                gabarits_features
            )
        else:
            # Pas de CV, utiliser seulement NLP et gabarits
            predicted_class = nlp_pred[0]
            confidence = nlp_pred[1]
            rejection_score = 1.0 - confidence
            result = (predicted_class, confidence, rejection_score, "nlp_only")
        
        predicted_class, confidence, rejection_score, strategy = result
        
        # Résultat final
        result_dict = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "rejection_score": float(rejection_score),
            "strategy": strategy,
            "cv": {
                "class": cv_pred[0] if cv_pred else None,
                "confidence": float(cv_pred[1]) if cv_pred else 0.0
            },
            "nlp": {
                "class": nlp_pred[0],
                "confidence": float(nlp_pred[1])
            },
            "gabarits_scores": {k: float(v) for k, v in gabarits_scores.items()}
        }
        
        return result_dict
    
    def classify_pdf(self, pdf_path: str) -> list:
        """
        Classifie un PDF (peut contenir plusieurs pages)
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Liste de résultats par page
        """
        # Convertir PDF en images
        image_paths = self.pdf_converter.convert_pdf(pdf_path)
        
        results = []
        for img_path in image_paths:
            try:
                result = self.classify_image(img_path)
                result["pdf_path"] = pdf_path
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur lors de la classification de {img_path}: {e}")
                results.append({
                    "image_path": img_path,
                    "pdf_path": pdf_path,
                    "error": str(e)
                })
        
        return results
    
    def process_directory(self, input_dir: str, output_dir: str):
        """
        Traite un dossier de PDFs
        
        Args:
            input_dir: Dossier d'entrée
            output_dir: Dossier de sortie
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Créer des sous-dossiers par classe
        for cls in CLASSES:
            (output_path / cls).mkdir(exist_ok=True)
        (output_path / "a_verifier").mkdir(exist_ok=True)
        
        # Lister les PDFs
        pdf_files = list(input_path.rglob("*.pdf"))
        
        logger.info(f"Traitement de {len(pdf_files)} PDF(s)...")
        
        all_results = []
        
        for pdf_file in tqdm(pdf_files, desc="Traitement"):
            try:
                results = self.classify_pdf(str(pdf_file))
                
                for result in results:
                    predicted_class = result.get("predicted_class")
                    confidence = result.get("confidence", 0.0)
                    rejection_score = result.get("rejection_score", 1.0)
                    
                    # Décision de classement
                    if rejection_score > 0.5 or confidence < 0.6:
                        target_dir = output_path / "a_verifier"
                    else:
                        target_dir = output_path / predicted_class
                    
                    # Copier le fichier
                    target_file = target_dir / pdf_file.name
                    import shutil
                    shutil.copy2(pdf_file, target_file)
                    
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Erreur avec {pdf_file}: {e}")
                all_results.append({
                    "pdf_path": str(pdf_file),
                    "error": str(e)
                })
        
        # Sauvegarder le rapport
        report_path = output_path / f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Traitement terminé. Rapport sauvegardé: {report_path}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Classification de documents administratifs")
    parser.add_argument("--input", "-i", required=True, help="Fichier PDF ou dossier d'entrée")
    parser.add_argument("--output", "-o", required=True, help="Dossier de sortie")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--light", action="store_true", help="Utiliser le modèle léger")
    
    args = parser.parse_args()
    
    # Initialiser le classificateur
    classifier = DocumentClassifier(device=args.device, use_light_model=args.light)
    
    # Traiter
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Fichier unique
        results = classifier.classify_pdf(str(input_path))
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif input_path.is_dir():
        # Dossier
        classifier.process_directory(str(input_path), args.output)
    else:
        logger.error(f"Entrée invalide: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

