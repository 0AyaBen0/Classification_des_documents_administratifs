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
        # Vérifier que les prédictions sont valides
        if not isinstance(cv_pred, tuple) or len(cv_pred) != 2:
            logger.warning(f"cv_pred invalide dans fuse_perfect_agreement: {cv_pred}")
            return None
        if not isinstance(nlp_pred, tuple) or len(nlp_pred) != 2:
            logger.warning(f"nlp_pred invalide dans fuse_perfect_agreement: {nlp_pred}")
            return None
        
        cv_class, cv_conf = cv_pred
        nlp_class, nlp_conf = nlp_pred
        
        # Accord parfait - seuil plus bas pour accepter plus souvent
        if cv_class == nlp_class and cv_conf > 0.6 and nlp_conf > 0.6:
            # Fusion multiplicative pour accord: confiance plus élevée
            final_conf = min(0.95, (cv_conf * nlp_conf) ** 0.5)  # Moyenne géométrique
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
        # Vérifier que cv_pred est valide
        if not isinstance(cv_pred, tuple) or len(cv_pred) != 2:
            logger.warning(f"cv_pred invalide dans fuse_cv_strong: {cv_pred}")
            return None
        
        cv_class, cv_conf = cv_pred
        
        # CV fort - seuil plus bas (0.7 au lieu de 0.9)
        if cv_conf > 0.7:
            # Vérifier la validation par gabarits (seuil plus bas aussi)
            if cv_class in gabarits_scores and gabarits_scores[cv_class] > 0.5:
                # Ajuster la confiance avec gabarits (fusion améliorée)
                gabarits_score = gabarits_scores[cv_class]
                # Fusion pondérée avec bonus si gabarits très forts
                if gabarits_score > 0.8:
                    final_conf = min(0.95, cv_conf * 0.6 + gabarits_score * 0.4)
                else:
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
        # Vérifier que nlp_pred est valide
        if not isinstance(nlp_pred, tuple) or len(nlp_pred) != 2:
            logger.warning(f"nlp_pred invalide dans fuse_nlp_strong: {nlp_pred}")
            return None
        
        nlp_class, nlp_conf = nlp_pred
        
        # NLP fort - seuil plus bas (0.7 au lieu de 0.9)
        if nlp_conf > 0.7:
            # Vérifier les motifs si disponibles
            if pattern_scores and nlp_class in pattern_scores:
                pattern_score = pattern_scores[nlp_class]
                if pattern_score > 0.5:  # Seuil plus bas
                    # Ajuster la confiance avec motifs (fusion améliorée)
                    if pattern_score > 0.8:
                        final_conf = min(0.95, nlp_conf * 0.7 + pattern_score * 0.3)
                    else:
                        final_conf = nlp_conf * 0.8 + pattern_score * 0.2
                    return (nlp_class, final_conf)
            else:
                # Pas de motifs, mais NLP confiant - accepter quand même
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
            try:
                if isinstance(nlp_pred, tuple) and len(nlp_pred) == 2:
                    nlp_class, nlp_conf = nlp_pred
                    if nlp_conf < 0.6:
                        return False, f"Facture nécessite une confiance NLP plus élevée"
                else:
                    logger.warning(f"nlp_pred invalide dans apply_business_rules: {nlp_pred} (type: {type(nlp_pred)}, len: {len(nlp_pred) if hasattr(nlp_pred, '__len__') else 'N/A'})")
            except Exception as e:
                logger.warning(f"Erreur lors de la vérification NLP dans apply_business_rules: {e}")
        
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
            # Vérifier que result est un tuple de 2 éléments
            if isinstance(result, tuple) and len(result) == 2:
                predicted_class, confidence = result
                is_valid, message = self.apply_business_rules(
                    predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
                )
                if is_valid:
                    return (predicted_class, confidence, 0.0, "accord_parfait")
            else:
                logger.warning(f"fuse_perfect_agreement a retourné un résultat invalide: {result}")
        
        # Stratégie 2: CV fort + gabarits
        result = self.fuse_cv_strong(cv_pred, nlp_pred, gabarits_scores)
        if result:
            # Vérifier que result est un tuple de 2 éléments
            if isinstance(result, tuple) and len(result) == 2:
                predicted_class, confidence = result
                is_valid, message = self.apply_business_rules(
                    predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
                )
                if is_valid:
                    return (predicted_class, confidence, 0.0, "cv_fort_gabarits")
            else:
                logger.warning(f"fuse_cv_strong a retourné un résultat invalide: {result}")
        
        # Stratégie 3: NLP fort + motifs
        result = self.fuse_nlp_strong(cv_pred, nlp_pred, pattern_scores)
        if result:
            # Vérifier que result est un tuple de 2 éléments
            if isinstance(result, tuple) and len(result) == 2:
                predicted_class, confidence = result
                is_valid, message = self.apply_business_rules(
                    predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
                )
                if is_valid:
                    return (predicted_class, confidence, 0.0, "nlp_fort_motifs")
            else:
                logger.warning(f"fuse_nlp_strong a retourné un résultat invalide: {result}")
        
        # Stratégie 4: Fusion pondérée standard
        # Vérifier que les prédictions sont valides
        if not isinstance(cv_pred, tuple) or len(cv_pred) != 2:
            logger.warning(f"cv_pred invalide dans fuse (stratégie 4): {cv_pred}")
            # Fallback: utiliser seulement NLP
            if isinstance(nlp_pred, tuple) and len(nlp_pred) == 2:
                nlp_class, nlp_conf = nlp_pred
                return (nlp_class, nlp_conf, 1.0 - nlp_conf, "nlp_only_fallback")
            else:
                # Fallback ultime
                return ("unknown", 0.0, 1.0, "error")
        
        if not isinstance(nlp_pred, tuple) or len(nlp_pred) != 2:
            logger.warning(f"nlp_pred invalide dans fuse (stratégie 4): {nlp_pred}")
            # Fallback: utiliser seulement CV
            cv_class, cv_conf = cv_pred
            return (cv_class, cv_conf, 1.0 - cv_conf, "cv_only_fallback")
        
        cv_class, cv_conf = cv_pred
        nlp_class, nlp_conf = nlp_pred
        
        # Calculer les scores pondérés avec fusion améliorée
        scores = {}
        all_classes = set([cv_class, nlp_class] + list(gabarits_scores.keys()))
        
        # Bonus si CV et NLP sont d'accord (fusion multiplicative pour accord)
        agreement_bonus = 1.5 if cv_class == nlp_class else 1.0
        
        # Bonus si gabarits valident la prédiction CV ou NLP
        for cls in all_classes:
            score = 0.0
            
            # CV - avec bonus d'accord
            if cls == cv_class:
                base_cv_score = cv_conf * self.cv_weight_norm
                if cv_class == nlp_class:
                    # Fusion multiplicative quand accord: CV * NLP
                    score += base_cv_score * agreement_bonus
                else:
                    score += base_cv_score
            
            # NLP - avec bonus d'accord
            if cls == nlp_class:
                base_nlp_score = nlp_conf * self.nlp_weight_norm
                if cv_class == nlp_class:
                    # Fusion multiplicative quand accord
                    score += base_nlp_score * agreement_bonus
                else:
                    score += base_nlp_score
            
            # Gabarits - avec bonus si valide CV ou NLP
            if cls in gabarits_scores:
                gabarit_score = gabarits_scores[cls]
                # Bonus si gabarits valident CV ou NLP
                if cls == cv_class or cls == nlp_class:
                    # Gabarits valident la prédiction
                    gabarit_bonus = 1.3
                else:
                    gabarit_bonus = 1.0
                score += gabarit_score * self.gabarits_weight * gabarit_bonus
            
            scores[cls] = score
        
        # Normaliser les scores pour qu'ils soient entre 0 et 1
        # Utiliser softmax pour une meilleure distribution
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            # Normalisation simple mais efficace
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {cls: score / total_score for cls, score in scores.items()}
            else:
                # Fallback: distribution uniforme
                scores = {cls: 1.0 / len(scores) for cls in scores}
        
        # Trouver la classe prédite (toujours choisir la meilleure)
        predicted_class = max(scores, key=scores.get) if scores else cv_class
        confidence = scores[predicted_class] if predicted_class in scores else 0.5
        
        # Calculer le score de rejet
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            rejection_score = max(0.0, 1.0 - gap - confidence)
        else:
            rejection_score = 1.0 - confidence
        
        # Validation métier (optionnelle - on classe quand même)
        is_valid, message = self.apply_business_rules(
            predicted_class, cv_pred, nlp_pred, gabarits_scores, image_features
        )
        
        # Pénalité réduite pour violation des règles (on classe quand même)
        if not is_valid:
            rejection_score += 0.15  # Pénalité réduite (au lieu de 0.3)
            confidence *= 0.9  # Légère réduction de confiance
        
        # Toujours retourner une prédiction (pas de rejet)
        # Le rejection_score sert juste d'indicateur de qualité
        strategy = "fusion_ponderee"
        if cv_class == nlp_class:
            strategy = "accord_parfait"
        elif cv_conf > 0.7 and predicted_class == cv_class:
            strategy = "cv_fort_gabarits"
        elif nlp_conf > 0.7 and predicted_class == nlp_class:
            strategy = "nlp_fort_motifs"
        
        return (predicted_class, confidence, rejection_score, strategy)
    
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

