"""
Configuration du logger
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "classification_docs",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure un logger
    
    Args:
        name: Nom du logger
        log_file: Fichier de log (optionnel)
        level: Niveau de log
        
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger configuré")

