"""
Script pour optimiser les seuils de rejection_score et confidence
Propose des seuils optimaux basés sur des principes statistiques
"""

import numpy as np
from typing import Dict, List, Tuple

def calculate_optimal_thresholds(
    target_auto_ratio: float = 0.8,
    conservative: bool = False
) -> Dict:
    """
    Calcule les seuils optimaux basés sur des principes statistiques
    
    Args:
        target_auto_ratio: Ratio cible d'images classées automatiquement (0.0 à 1.0)
        conservative: Si True, privilégie la qualité (plus de vérifications)
    
    Returns:
        Dict avec les seuils recommandés et explications
    """
    
    if conservative:
        # Mode conservateur : privilégie la qualité
        # Rejection plus bas = plus d'images en vérification
        # Confidence plus haut = besoin de plus de confiance
        rejection_threshold = 0.4  # Plus bas = plus strict
        confidence_threshold = 0.7  # Plus haut = plus strict
        description = "Mode conservateur : privilégie la qualité, plus d'images en vérification"
    else:
        # Mode équilibré : bon compromis
        rejection_threshold = 0.6  # Moyen
        confidence_threshold = 0.55  # Moyen
        description = "Mode équilibré : bon compromis entre qualité et automatisation"
    
    # Mode permissif
    if target_auto_ratio > 0.85:
        rejection_threshold = 0.7  # Plus haut = plus permissif
        confidence_threshold = 0.5  # Plus bas = plus permissif
        description = "Mode permissif : plus d'automatisation, moins de vérifications"
    
    return {
        'rejection_threshold': rejection_threshold,
        'confidence_threshold': confidence_threshold,
        'description': description,
        'expected_auto_ratio': target_auto_ratio
    }

def get_recommendations() -> List[Dict]:
    """Retourne plusieurs recommandations de seuils"""
    recommendations = []
    
    # 1. Conservateur (haute qualité)
    rec1 = calculate_optimal_thresholds(target_auto_ratio=0.6, conservative=True)
    rec1['name'] = 'Conservateur (Haute Qualité)'
    rec1['use_case'] = 'Production critique, erreurs coûteuses'
    recommendations.append(rec1)
    
    # 2. Équilibré
    rec2 = calculate_optimal_thresholds(target_auto_ratio=0.75, conservative=False)
    rec2['name'] = 'Équilibré (Recommandé)'
    rec2['use_case'] = 'Usage général, bon compromis'
    recommendations.append(rec2)
    
    # 3. Permissif
    rec3 = calculate_optimal_thresholds(target_auto_ratio=0.9, conservative=False)
    rec3['name'] = 'Permissif (Plus d\'Automatisation)'
    rec3['use_case'] = 'Volume élevé, erreurs acceptables'
    recommendations.append(rec3)
    
    return recommendations

def print_recommendations():
    """Affiche les recommandations"""
    print("=" * 80)
    print("OPTIMISATION DES SEUILS DE CLASSIFICATION")
    print("=" * 80)
    print("\nLes images sont placées dans 'a_verifier' si:")
    print("  - rejection_score > rejection_threshold OU")
    print("  - confidence < confidence_threshold")
    print("\n" + "=" * 80)
    print("RECOMMANDATIONS")
    print("=" * 80)
    
    recommendations = get_recommendations()
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   {rec['description']}")
        print(f"   Cas d'usage: {rec['use_case']}")
        print(f"   rejection_threshold: {rec['rejection_threshold']:.3f}")
        print(f"   confidence_threshold: {rec['confidence_threshold']:.3f}")
        print(f"   Ratio attendu d'auto-classification: {rec['expected_auto_ratio']*100:.0f}%")
    
    # Recommandation principale
    balanced = recommendations[1]  # Équilibré
    print("\n" + "=" * 80)
    print("RECOMMANDATION PRINCIPALE (ÉQUILIBRÉ)")
    print("=" * 80)
    print(f"\n✅ rejection_threshold = {balanced['rejection_threshold']:.3f}")
    print(f"✅ confidence_threshold = {balanced['confidence_threshold']:.3f}")
    
    print("\n" + "=" * 80)
    print("COMMENT APPLIQUER")
    print("=" * 80)
    print("\n1. Dans main.py, ligne 496, remplacez:")
    print("   if rejection_score > 0.5 or confidence < 0.6:")
    print("\n   Par:")
    print(f"   if rejection_score > {balanced['rejection_threshold']:.3f} or confidence < {balanced['confidence_threshold']:.3f}:")
    
    print("\n2. OU dans config.yaml, section fusion, ajoutez/modifiez:")
    print(f"   fusion:")
    print(f"     rejection_threshold: {balanced['rejection_threshold']:.3f}")
    print(f"     confidence_threshold: {balanced['confidence_threshold']:.3f}")
    
    print("\n" + "=" * 80)
    print("EXPLICATION DES SEUILS")
    print("=" * 80)
    print("\n📊 rejection_score:")
    print("   - Mesure l'ambiguïté entre les classes")
    print("   - Plus bas = plus strict (plus d'images en vérification)")
    print("   - Plus haut = plus permissif (moins d'images en vérification)")
    print("   - Recommandé: 0.5-0.7")
    
    print("\n📈 confidence:")
    print("   - Mesure la confiance dans la prédiction")
    print("   - Plus haut = plus strict (besoin de plus de confiance)")
    print("   - Plus bas = plus permissif (accepte moins de confiance)")
    print("   - Recommandé: 0.5-0.7")
    
    print("\n💡 Astuce:")
    print("   - Si trop d'images dans 'a_verifier': augmentez rejection_threshold ou baissez confidence_threshold")
    print("   - Si trop d'erreurs: baissez rejection_threshold ou augmentez confidence_threshold")

def create_threshold_config():
    """Crée un fichier de configuration avec les seuils optimisés"""
    balanced = get_recommendations()[1]
    
    config_content = f"""# Configuration des seuils de classification optimisés
# Généré automatiquement par optimize_thresholds.py

# Seuils recommandés (mode équilibré)
rejection_threshold: {balanced['rejection_threshold']:.3f}
confidence_threshold: {balanced['confidence_threshold']:.3f}

# Explication:
# - rejection_threshold: Seuil pour le score de rejet (0.0-1.0)
#   Plus bas = plus strict = plus d'images en vérification
#   Plus haut = plus permissif = moins d'images en vérification
#
# - confidence_threshold: Seuil pour la confiance (0.0-1.0)
#   Plus haut = plus strict = besoin de plus de confiance
#   Plus bas = plus permissif = accepte moins de confiance

# Pour ajuster:
# - Si trop d'images dans 'a_verifier': augmentez rejection_threshold ou baissez confidence_threshold
# - Si trop d'erreurs: baissez rejection_threshold ou augmentez confidence_threshold
"""
    
    with open('thresholds_config.txt', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("\n✅ Fichier 'thresholds_config.txt' créé avec les seuils recommandés")

if __name__ == "__main__":
    print_recommendations()
    create_threshold_config()
