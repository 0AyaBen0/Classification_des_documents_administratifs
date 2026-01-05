"""
Classification par motifs sémantiques
Détection de mots-clés et expressions caractéristiques
"""

import re
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticPatternMatcher:
    """Matcher de motifs sémantiques pour classification"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le matcher de motifs
        
        Args:
            config_path: Chemin vers le fichier de configuration des gabarits
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "models" / "gabarits" / "gabarits_config.json"
        
        self.config_path = Path(config_path)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict]:
        """Charge les motifs depuis la configuration"""
        if not self.config_path.exists():
            logger.warning(f"Fichier de configuration non trouvé: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        patterns = {}
        for family_name, family_config in config.items():
            keywords = family_config.get('keywords', [])
            
            # Créer des patterns regex
            patterns_list = []
            for keyword in keywords:
                # Pattern insensible à la casse et aux accents
                pattern = self._create_flexible_pattern(keyword)
                patterns_list.append(pattern)
            
            patterns[family_name] = {
                'keywords': keywords,
                'patterns': patterns_list,
                'weights': {kw: 1.0 for kw in keywords}  # Poids par défaut
            }
        
        logger.info(f"Patterns chargés pour {len(patterns)} familles")
        return patterns
    
    def _create_flexible_pattern(self, keyword: str) -> re.Pattern:
        """
        Crée un pattern regex flexible (insensible à la casse, accents)
        
        Args:
            keyword: Mot-clé
            
        Returns:
            Pattern regex compilé
        """
        # Normaliser les accents
        normalized = keyword.lower()
        
        # Échapper les caractères spéciaux
        escaped = re.escape(normalized)
        
        # Pattern insensible à la casse
        pattern = f"(?i){escaped}"
        
        return re.compile(pattern)
    
    def match_patterns(self, text: str) -> Dict[str, float]:
        """
        Calcule les scores de correspondance pour chaque famille
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire {nom_famille: score}
        """
        text_lower = text.lower()
        scores = {}
        
        for family_name, family_patterns in self.patterns.items():
            total_score = 0.0
            matches_count = 0
            
            for keyword, pattern in zip(family_patterns['keywords'], family_patterns['patterns']):
                # Compter les occurrences
                matches = pattern.findall(text_lower)
                count = len(matches)
                
                if count > 0:
                    weight = family_patterns['weights'].get(keyword, 1.0)
                    # Score basé sur le nombre d'occurrences (avec saturation)
                    keyword_score = min(1.0, count * 0.2) * weight
                    total_score += keyword_score
                    matches_count += count
            
            # Normaliser le score
            if len(family_patterns['keywords']) > 0:
                normalized_score = total_score / len(family_patterns['keywords'])
            else:
                normalized_score = 0.0
            
            # Bonus pour le nombre total de matches
            if matches_count > 0:
                normalized_score = min(1.0, normalized_score * (1 + matches_count * 0.1))
            
            scores[family_name] = normalized_score
        
        return scores
    
    def find_keywords(self, text: str, family: str) -> List[Tuple[str, int]]:
        """
        Trouve tous les mots-clés d'une famille dans le texte
        
        Args:
            text: Texte à analyser
            family: Nom de la famille
            
        Returns:
            Liste de tuples (mot-clé, nombre d'occurrences)
        """
        if family not in self.patterns:
            return []
        
        text_lower = text.lower()
        results = []
        
        for keyword, pattern in zip(
            self.patterns[family]['keywords'],
            self.patterns[family]['patterns']
        ):
            matches = pattern.findall(text_lower)
            count = len(matches)
            if count > 0:
                results.append((keyword, count))
        
        return results
    
    def get_confidence(self, scores: Dict[str, float]) -> Tuple[str, float, float]:
        """
        Calcule la confiance de la prédiction
        
        Args:
            scores: Scores pour chaque famille
            
        Returns:
            Tuple (classe_prédite, score, confiance)
        """
        if not scores:
            return None, 0.0, 0.0
        
        # Trouver la classe avec le score le plus élevé
        predicted_class = max(scores, key=scores.get)
        predicted_score = scores[predicted_class]
        
        # Calculer la confiance (écart entre le meilleur et le second)
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, gap * 2 + predicted_score * 0.5)
        else:
            confidence = predicted_score
        
        return predicted_class, predicted_score, confidence


if __name__ == "__main__":
    matcher = SemanticPatternMatcher()
    print("SemanticPatternMatcher initialisé")
    print(f"Familles configurées: {list(matcher.patterns.keys())}")

