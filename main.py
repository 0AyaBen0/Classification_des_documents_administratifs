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
            model_path = Path("models/cv/best_model.pth")
            if model_path.exists():
                # Détecter le type de modèle depuis le checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Détecter si c'est un modèle léger (MobileNet) ou standard (ResNet50)
                # Les modèles légers ont "backbone.features" au lieu de "backbone.conv1"
                has_mobilenet_features = any('backbone.features' in key for key in state_dict.keys())
                has_resnet_features = any('backbone.conv1' in key or 'backbone.layer1' in key for key in state_dict.keys())
                
                # Prioriser use_light_model si spécifié, sinon détecter depuis le checkpoint
                if use_light_model or (has_mobilenet_features and not has_resnet_features):
                    from src.computer_vision.hybrid_cv_model import LightHybridCVModel
                    # Utiliser hidden_dim=32 pour correspondre au modèle sauvegardé
                    self.cv_model = LightHybridCVModel(
                        num_classes=len(CLASSES), 
                        gabarit_features_dim=10,
                        hidden_dim=32  # Correspond au modèle sauvegardé
                    )
                    logger.info("Utilisation du modèle léger (MobileNet)")
                else:
                    self.cv_model = HybridCVModel(num_classes=len(CLASSES), gabarit_features_dim=10)
                    logger.info("Utilisation du modèle standard (ResNet50)")
                
                # Charger les poids
                try:
                    self.cv_model.load_state_dict(state_dict, strict=False)
                    logger.info("Modèle CV chargé depuis best_model.pth")
                except Exception as load_error:
                    logger.warning(f"Impossible de charger tous les poids: {load_error}")
                    logger.warning("Utilisation du modèle pré-entraîné uniquement")
            else:
                # Pas de checkpoint, créer un nouveau modèle
                if use_light_model:
                    from src.computer_vision.hybrid_cv_model import LightHybridCVModel
                    self.cv_model = LightHybridCVModel(num_classes=len(CLASSES), gabarit_features_dim=10)
                else:
                    self.cv_model = HybridCVModel(num_classes=len(CLASSES), gabarit_features_dim=10)
                logger.warning("Modèle CV non entraîné. Utilisation du modèle pré-entraîné uniquement.")
            
            if self.cv_model is not None:
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
        try:
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
                        try:
                            output = self.cv_model.predict(img_tensor, gabarit_tensor)
                            # Vérifier que predict retourne bien 2 valeurs
                            if isinstance(output, tuple) and len(output) == 2:
                                probs, preds = output
                            else:
                                raise ValueError(f"predict doit retourner (probs, preds), reçu: {type(output)}")
                        except Exception as predict_error:
                            logger.error(f"Erreur lors de l'appel à predict: {predict_error}")
                            raise
                    
                    cv_class_idx = preds[0].item()
                    cv_class = CLASSES[cv_class_idx]
                    cv_conf = probs[0][cv_class_idx].item()
                    cv_pred = (cv_class, cv_conf)
                except Exception as e:
                    logger.error(f"Erreur classification CV: {e}")
                    cv_pred = None
            
            # 3. Classification NLP
            nlp_pred = None
            nlp_scores = {}
            try:
                nlp_result = self.nlp_classifier.classify_image(image, CLASSES)
                # Vérifier que le résultat est un tuple de 3 éléments
                if isinstance(nlp_result, tuple) and len(nlp_result) == 3:
                    nlp_class, nlp_conf, nlp_scores = nlp_result
                    nlp_pred = (nlp_class, nlp_conf) if nlp_class else None
                else:
                    logger.warning(f"classify_image() a retourné un résultat inattendu: {nlp_result}")
                    nlp_pred = None
                    nlp_scores = {}
            except Exception as e:
                logger.error(f"Erreur lors de la classification NLP: {e}")
                nlp_pred = None
                nlp_scores = {}
            
            if nlp_pred is None:
                # Fallback: utiliser les motifs sémantiques uniquement
                try:
                    text = self.nlp_classifier.extract_text(image)
                    if text:
                        pattern_scores = self.nlp_classifier.pattern_matcher.match_patterns(text)
                        if pattern_scores:
                            nlp_class = max(pattern_scores, key=pattern_scores.get)
                            nlp_conf = pattern_scores[nlp_class]
                            nlp_pred = (nlp_class, nlp_conf)
                            nlp_scores = {'patterns': pattern_scores}
                except Exception as e:
                    logger.warning(f"Erreur lors du fallback NLP: {e}")
            
            if nlp_pred is None:
                nlp_pred = (CLASSES[0], 0.0)  # Fallback
                nlp_scores = {}
            
            # 4. Fusion multimodale
            result = None
            try:
                # Vérifier et normaliser cv_pred
                if cv_pred:
                    if not isinstance(cv_pred, tuple) or len(cv_pred) != 2:
                        logger.warning(f"cv_pred invalide (type: {type(cv_pred)}, valeur: {cv_pred}), utilisation de fallback")
                        cv_pred = None
                    else:
                        logger.debug(f"cv_pred valide: {cv_pred}")
                
                # Vérifier et normaliser nlp_pred
                if nlp_pred:
                    if not isinstance(nlp_pred, tuple) or len(nlp_pred) != 2:
                        logger.warning(f"nlp_pred invalide (type: {type(nlp_pred)}, valeur: {nlp_pred}), utilisation de fallback")
                        nlp_pred = (CLASSES[0], 0.0)
                    else:
                        logger.debug(f"nlp_pred valide: {nlp_pred}")
                else:
                    nlp_pred = (CLASSES[0], 0.0)
                
                if cv_pred and nlp_pred:
                    pattern_scores = nlp_scores.get('patterns', {}) if isinstance(nlp_scores, dict) else {}
                    try:
                        logger.debug(f"Appel de fuse() avec cv_pred={cv_pred}, nlp_pred={nlp_pred}")
                        result = self.fusion.fuse(
                            cv_pred,
                            nlp_pred,
                            gabarits_scores,
                            pattern_scores,
                            gabarits_features
                        )
                        logger.debug(f"fuse() a retourné: {result} (type: {type(result)})")
                        # Vérifier que result est un tuple de 4 éléments
                        if not isinstance(result, tuple) or len(result) != 4:
                            raise ValueError(f"fuse() a retourné un résultat invalide: {result} (type: {type(result)}, len: {len(result) if hasattr(result, '__len__') else 'N/A'})")
                    except Exception as e:
                        logger.error(f"Erreur lors de la fusion: {e}", exc_info=True)
                        # Fallback: utiliser NLP uniquement
                        if isinstance(nlp_pred, tuple) and len(nlp_pred) == 2:
                            predicted_class = nlp_pred[0]
                            confidence = nlp_pred[1]
                            rejection_score = 1.0 - confidence
                            result = (predicted_class, confidence, rejection_score, "fusion_error")
                        else:
                            result = (CLASSES[0], 0.0, 1.0, "fusion_error")
                
                if result is None:
                    if nlp_pred and isinstance(nlp_pred, tuple) and len(nlp_pred) == 2:
                        # Pas de CV, utiliser seulement NLP et gabarits
                        predicted_class = nlp_pred[0]
                        confidence = nlp_pred[1]
                        rejection_score = 1.0 - confidence
                        result = (predicted_class, confidence, rejection_score, "nlp_only")
                    else:
                        # Fallback ultime: utiliser la première classe avec confiance minimale
                        predicted_class = CLASSES[0]
                        confidence = 0.0
                        rejection_score = 1.0
                        result = (predicted_class, confidence, rejection_score, "fallback")
                
                # Vérification finale avant décompression
                if not isinstance(result, tuple):
                    logger.error(f"Résultat n'est pas un tuple: {result} (type: {type(result)})")
                    result = (CLASSES[0], 0.0, 1.0, "error")
                elif len(result) != 4:
                    logger.error(f"Résultat n'a pas 4 éléments: {result} (len: {len(result)})")
                    # Essayer de corriger
                    if len(result) == 1:
                        result = (result[0] if isinstance(result[0], str) else CLASSES[0], 0.0, 1.0, "error")
                    elif len(result) == 2:
                        result = (result[0] if isinstance(result[0], str) else CLASSES[0], 
                                 result[1] if isinstance(result[1], (int, float)) else 0.0, 
                                 1.0, "error")
                    elif len(result) == 3:
                        result = (result[0] if isinstance(result[0], str) else CLASSES[0],
                                 result[1] if isinstance(result[1], (int, float)) else 0.0,
                                 result[2] if isinstance(result[2], (int, float)) else 1.0,
                                 "error")
                    else:
                        result = (CLASSES[0], 0.0, 1.0, "error")
                
                predicted_class, confidence, rejection_score, strategy = result
                
            except Exception as e:
                logger.error(f"Erreur critique lors de la fusion: {e}", exc_info=True)
                # Fallback de sécurité
                predicted_class = CLASSES[0]
                confidence = 0.0
                rejection_score = 1.0
                strategy = "error"
            
            # Résultat final
            # Vérifier cv_pred avant d'accéder
            cv_class = None
            cv_conf = 0.0
            if cv_pred:
                if isinstance(cv_pred, tuple) and len(cv_pred) >= 2:
                    cv_class = cv_pred[0]
                    cv_conf = float(cv_pred[1])
                else:
                    logger.warning(f"cv_pred invalide lors de la construction du résultat: {cv_pred}")
            
            # Vérifier nlp_pred avant d'accéder
            nlp_class = CLASSES[0]
            nlp_conf = 0.0
            if nlp_pred:
                if isinstance(nlp_pred, tuple) and len(nlp_pred) >= 2:
                    nlp_class = nlp_pred[0]
                    nlp_conf = float(nlp_pred[1])
                else:
                    logger.warning(f"nlp_pred invalide lors de la construction du résultat: {nlp_pred}")
                    nlp_class = CLASSES[0]
                    nlp_conf = 0.0
            
            result_dict = {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "rejection_score": float(rejection_score),
                "strategy": strategy,
                "cv": {
                    "class": cv_class,
                    "confidence": cv_conf
                },
                "nlp": {
                    "class": nlp_class,
                    "confidence": nlp_conf
                },
                "gabarits_scores": {k: float(v) for k, v in gabarits_scores.items()}
            }
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Erreur critique dans classify_image pour {image_path}: {e}", exc_info=True)
            # Retourner un résultat d'erreur
            return {
                "image_path": image_path,
                "error": str(e),
                "predicted_class": CLASSES[0],
                "confidence": 0.0,
                "rejection_score": 1.0,
                "strategy": "error",
                "cv": {"class": None, "confidence": 0.0},
                "nlp": {"class": CLASSES[0], "confidence": 0.0},
                "gabarits_scores": {}
            }
    
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
        Traite un dossier de PDFs ou d'images
        
        Args:
            input_dir: Dossier d'entrée (PDFs ou images)
            output_dir: Dossier de sortie
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Créer des sous-dossiers par classe
        for cls in CLASSES:
            (output_path / cls).mkdir(exist_ok=True)
        # Plus de dossier a_verifier - on classe toujours dans une des 5 classes
        
        # Détecter le type de fichiers (PDFs ou images)
        pdf_files = list(input_path.rglob("*.pdf"))
        image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.jpeg"))
        
        all_results = []
        
        # Traiter les PDFs
        if pdf_files:
            logger.info(f"Traitement de {len(pdf_files)} PDF(s)...")
            
            for pdf_file in tqdm(pdf_files, desc="Traitement PDFs"):
                try:
                    results = self.classify_pdf(str(pdf_file))
                    
                    for result in results:
                        predicted_class = result.get("predicted_class")
                        confidence = result.get("confidence", 0.0)
                        rejection_score = result.get("rejection_score", 1.0)
                        
                        # Toujours classer dans une des 5 classes (pas de a_verifier)
                        # Utiliser la classe prédite même si la confiance est basse
                        if predicted_class and predicted_class in CLASSES:
                            target_dir = output_path / predicted_class
                        else:
                            # Fallback: utiliser la première classe si prédiction invalide
                            target_dir = output_path / CLASSES[0]
                            logger.warning(f"Classe prédite invalide: {predicted_class}, utilisation de {CLASSES[0]}")
                        
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
        
        # Traiter les images
        if image_files:
            logger.info(f"Traitement de {len(image_files)} image(s)...")
            
            for img_file in tqdm(image_files, desc="Traitement Images"):
                try:
                    result = self.classify_image(str(img_file))
                    
                    predicted_class = result.get("predicted_class")
                    confidence = result.get("confidence", 0.0)
                    rejection_score = result.get("rejection_score", 1.0)
                    
                    # Toujours classer dans une des 5 classes (pas de a_verifier)
                    # Utiliser la classe prédite même si la confiance est basse
                    if predicted_class and predicted_class in CLASSES:
                        target_dir = output_path / predicted_class
                    else:
                        # Fallback: utiliser la première classe si prédiction invalide
                        target_dir = output_path / CLASSES[0]
                        logger.warning(f"Classe prédite invalide: {predicted_class}, utilisation de {CLASSES[0]}")
                    
                    # Copier l'image
                    target_file = target_dir / img_file.name
                    import shutil
                    shutil.copy2(img_file, target_file)
                    
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Erreur avec {img_file}: {e}")
                    all_results.append({
                        "image_path": str(img_file),
                        "error": str(e)
                    })
        
        if not pdf_files and not image_files:
            logger.warning(f"Aucun fichier PDF ou image trouvé dans {input_dir}")
            return
        
        # Sauvegarder le rapport
        report_path = output_path / f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Traitement terminé. Rapport sauvegardé: {report_path}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Classification de documents administratifs")
    parser.add_argument("--input", "-i", required=True, help="Fichier PDF/image ou dossier d'entrée (PDFs ou images PNG/JPG)")
    parser.add_argument("--output", "-o", required=True, help="Dossier de sortie")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--light", action="store_true", help="Utiliser le modèle léger")
    
    args = parser.parse_args()
    
    # Initialiser le classificateur
    classifier = DocumentClassifier(device=args.device, use_light_model=args.light)
    
    # Traiter
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Fichier unique
        if input_path.suffix.lower() == ".pdf":
            # Fichier PDF
            results = classifier.classify_pdf(str(input_path))
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            # Fichier image
            result = classifier.classify_image(str(input_path))
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            logger.error(f"Format de fichier non supporté: {input_path.suffix}")
            logger.error("Formats supportés: .pdf, .png, .jpg, .jpeg")
            sys.exit(1)
    elif input_path.is_dir():
        # Dossier (peut contenir PDFs et/ou images)
        classifier.process_directory(str(input_path), args.output)
    else:
        logger.error(f"Entrée invalide: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

