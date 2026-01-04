"""
Fusion Multimodale Intelligente
Combine CV, NLP et Gabarits avec validation métier
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusion:
    """Système de fusion multimodale intelligent"""
    
    def __init__(
        self,
        cv_weight: float = 0.5,
        nlp_weight: float = 0.5,
        gabarits_weight: float = 0.3,
        confidence_threshold: float = 0.8,
        rejection_threshold: float = 0.5
    ):
        """
        Initialise le système de fusion
        
        Args:
            cv_weight: Poids pour Computer Vision
            nlp_weight: Poids pour NLP
            gabarits_weight: Poids pour les gabarits
            confidence_threshold: Seuil de confiance pour acceptation
            rejection_threshold: Seuil de rejet
        """
        self.cv_weight = cv_weight
        self.nlp_weight = nlp_weight
        self.gabarits_weight = gabarits_weight
        self.confidence_threshold = confidence_threshold
        self.rejection_threshold = rejection_threshold
        
        # Normaliser les poids
        total_weight = cv_weight + nlp_weight
        if total_weight > 0:
            self.cv_weight_norm = cv_weight / total_weight
            self.nlp_weight_norm = nlp_weight / total_weight
        else:
            self.cv_weight_norm = 0.5
            self.nlp_weight_norm = 0.5
    
    def fuse_perfect_agreement(
        self,
        cv_pred: Tuple[str, float],
        nlp_pred: Tuple[str, float]
    ) -> Optional[Tuple[str, float]]:
        """
        Stratégie 1: Accord parfait
        
        Si CV et NLP donnent la même classe avec haute confiance, accepter
        
        Args:
            cv_pred: Tuple (classe, confiance) de CV
            nlp_pred: Tuple (classe, confiance) de NLP
            
        Returns:
            Tuple (classe, confiance) ou None si pas d'accord
        """
        cv_class, cv_conf = cv_pred
        nlp_class, nlp_conf = nlp_pred
        
        if cv_class == nlp_class and cv_conf > self.confidence_threshold and nlp_conf > self.confidence_threshold:
            # Confiance moyenne
            final_conf = (cv_conf + nlp_conf) / 2
            return (cv_class, final_conf)
        
        return None
    
    def fuse_cv_strong(
        self,
        cv_pred: Tuple[str, float],
        nlp_pred: Tuple[str, float],
        gabarits_scores: Dict[str, float]
    ) -> Optional[Tuple[str, float]]:
        """
        Stratégie 2: CV fort + validation gabarits
        
        Si CV est très confiant et gabarits valident, favoriser CV
        
        Args:
            cv_pred: Tuple (classe, confiance) de CV
            nlp_pred: Tuple (classe, confiance) de NLP
            gabarits_scores: Scores des gabarits
            
        Returns:
            Tuple (classe, confiance) ou None
        """
        cv_class, cv_conf = cv_pred
        
        if cv_conf > 0.9:
            # Vérifier la validation par gabarits
            if cv_class in gabarits_scores and gabarits_scores[cv_class] > 0.7:
                # Ajuster la confiance avec gabarits
                gabarits_score = gabarits_scores[cv_class]
                final_conf = cv_conf * 0.7 + gabarits_score * 0.3
                return (cv_class, final_conf)
        
        return None
    
    def fuse_nlp_strong(
        self,
        cv_pred: Tuple[str, float],
        nlp_pred: Tuple[str, float],
        pattern_scores: Optional[Dict[str, float]] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Stratégie 3: NLP fort + motifs textuels
        
        Si NLP est très confiant et motifs valident, favoriser NLP
        
        Args:
            cv_pred: Tuple (classe, confiance) de CV
            nlp_pred: Tuple (classe, confiance) de NLP
            pattern_scores: Scores des motifs sémantiques
            
        Returns:
            Tuple (classe, confiance) ou None
        """
        nlp_class, nlp_conf = nlp_pred
        
        if nlp_conf > 0.9:
            # Vérifier les motifs si disponibles
            if pattern_scores and nlp_class in pattern_scores:
                pattern_score = pattern_scores[nlp_class]
                if pattern_score > 0.6:
                    # Ajuster la confiance avec motifs
                    final_conf = nlp_conf * 0.8 + pattern_score * 0.2
                    return (nlp_class, final_conf)
            else:
                # Pas de motifs, mais NLP très confiant
                return (nlp_class, nlp_conf)
        
        return None
    
    def apply_business_rules(
        self,
        predicted_class: str,
        cv_pred: Tuple[str, float],
        nlp_pred: Tuple[str, float],
        gabarits_scores: Dict[str, float],
        image_features: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Applique les règles métier pour validation
        
        Args:
            predicted_class: Classe prédite
            cv_pred: Prédiction CV
            nlp_pred: Prédiction NLP
            gabarits_scores: Scores gabarits
            image_features: Features de l'image (optionnel)
            
        Returns:
            Tuple (valide, message)
        """
        # Règle 1: Pièce d'identité nécessite photo
        if predicted_class == "identite":
            if image_features:
                has_photo = image_features.get('photo_detection', 0) > 0.5
                if not has_photo:
                    return False, "Pièce d'identité nécessite une photo détectée"
            
            # Vérifier le ratio d'aspect
            aspect_ratio = gabarits_scores.get('aspect_ratio', 0)
            if aspect_ratio < 0.6 or aspect_ratio > 0.7:
                return False, "Ratio d'aspect incompatible avec une carte d'identité"
        
        # Règle 2: Relevé bancaire nécessite structure tabulaire
        if predicted_class == "releve_bancaire":
            tabular = gabarits_scores.get('tabular_structure', 0)
            if tabular < 0.5:
                return False, "Relevé bancaire nécessite une structure tabulaire"
        
        # Règle 3: Factures nécessitent unités de mesure
        if predicted_class in ["facture_electricite", "facture_eau"]:
            # Vérifier via NLP (unités kWh ou m³)
            nlp_class, nlp_conf = nlp_pred
            if nlp_conf < 0.6:
                return False, f"Facture nécessite une confiance NLP plus élevée"
        
        # Règle 4: Document employeur nécessite signature
        if predicted_class == "document_employeur":
            signature = gabarits_scores.get('signature_zone', 0)
            if signature < 0.3:
                return False, "Document employeur nécessite une zone de signature"
        
        return True, "Validation OK"
    
    def fuse(
        self,
        cv_pred: Tuple[str, float],
        nlp_pred: Tuple[str, float],
        gabarits_scores: Dict[str, float],
        pattern_scores: Optional[Dict[str, float]] = None,
        image_features: Optional[Dict] = None
    ) -> Tuple[str, float, float, str]:
        """
        Fusion principale avec toutes les stratégies
        
        Args:
            cv_pred: Prédiction CV (classe, confiance)
            nlp_pred: Prédiction NLP (classe, confiance)
            gabarits_scores: Scores des gabarits
            pattern_scores: Scores des motifs sémantiques
            image_features: Features de l'image
            
        Returns:
            Tuple (classe_prédite, confiance, score_rejet, stratégie_utilisée)
        """
        # Essayer les stratégies dans l'ordre
        
        # Stratégie 1: Accord parfait
        result = self.fuse_perfect_agreement(cv_pred, nlp_pred)
        if result:
            predicted_class, confidence = result
            is_valid, message = self.apply_business_rules(
                predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
            )
            if is_valid:
                return (predicted_class, confidence, 0.0, "accord_parfait")
        
        # Stratégie 2: CV fort + gabarits
        result = self.fuse_cv_strong(cv_pred, nlp_pred, gabarits_scores)
        if result:
            predicted_class, confidence = result
            is_valid, message = self.apply_business_rules(
                predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
            )
            if is_valid:
                return (predicted_class, confidence, 0.0, "cv_fort_gabarits")
        
        # Stratégie 3: NLP fort + motifs
        result = self.fuse_nlp_strong(cv_pred, nlp_pred, pattern_scores)
        if result:
            predicted_class, confidence = result
            is_valid, message = self.apply_business_rules(
                predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
            )
            if is_valid:
                return (predicted_class, confidence, 0.0, "nlp_fort_motifs")
        
        # Stratégie 4: Fusion pondérée standard
        cv_class, cv_conf = cv_pred
        nlp_class, nlp_conf = nlp_pred
        
        # Calculer les scores pondérés
        scores = {}
        classes = set([cv_class, nlp_class] + list(gabarits_scores.keys()))
        
        for cls in classes:
            score = 0.0
            
            # CV
            if cls == cv_class:
                score += cv_conf * self.cv_weight_norm
            
            # NLP
            if cls == nlp_class:
                score += nlp_conf * self.nlp_weight_norm
            
            # Gabarits
            if cls in gabarits_scores:
                score += gabarits_scores[cls] * self.gabarits_weight * 0.3
            
            scores[cls] = score
        
        # Trouver la classe prédite
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        # Calculer le score de rejet
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            rejection_score = max(0.0, 1.0 - gap - confidence)
        else:
            rejection_score = 1.0 - confidence
        
        # Validation métier
        is_valid, message = self.apply_business_rules(
            predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
        )
        
        if not is_valid:
            rejection_score += 0.3  # Pénalité pour violation des règles
        
        # Décision finale
        if confidence < self.rejection_threshold or rejection_score > 0.5:
            return (predicted_class, confidence, rejection_score, "rejet_manuel")
        
        return (predicted_class, confidence, rejection_score, "fusion_ponderee")
    
    def log_decision(
        self,
        decision: Tuple[str, float, float, str],
        cv_pred: Tuple[str, float],
        nlp_pred: Tuple[str, float],
        log_file: Optional[str] = None
    ):
        """
        Log une décision pour monitoring
        
        Args:
            decision: Tuple de décision
            cv_pred: Prédiction CV
            nlp_pred: Prédiction NLP
            log_file: Fichier de log (optionnel)
        """
        predicted_class, confidence, rejection_score, strategy = decision
        cv_class, cv_conf = cv_pred
        nlp_class, nlp_conf = nlp_pred
        
        log_entry = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "rejection_score": float(rejection_score),
            "strategy": strategy,
            "cv": {"class": cv_class, "confidence": float(cv_conf)},
            "nlp": {"class": nlp_class, "confidence": float(nlp_conf)}
        }
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ajouter au fichier JSON
            if log_path.exists():
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        else:
            logger.info(f"Décision: {predicted_class} (conf: {confidence:.3f}, rejet: {rejection_score:.3f}, strat: {strategy})")


if __name__ == "__main__":
    fusion = MultimodalFusion()
    print("MultimodalFusion initialisé")
    
    # Test
    cv_pred = ("identite", 0.85)
    nlp_pred = ("identite", 0.90)
    gabarits_scores = {"identite": 0.75, "releve_bancaire": 0.20}
    
    result = fusion.fuse(cv_pred, nlp_pred, gabarits_scores)
    print(f"Résultat: {result}")

