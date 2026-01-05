"""
Chargeur de configuration
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Fichier de configuration non trouvé: {config_path}")
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration chargée depuis {config_path}")
    return config


if __name__ == "__main__":
    config = load_config()
    print("Configuration chargée")

