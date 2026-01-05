"""
Module 1 : Configuration Offline
Gestionnaire de modèles pour fonctionnement complètement offline
"""

import os
import json
import hashlib
import torch
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineModelManager:
    """Gestionnaire de modèles pour fonctionnement offline"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Sous-dossiers pour chaque type de modèle
        self.cv_dir = self.models_dir / "cv"
        self.nlp_dir = self.models_dir / "nlp"
        self.gabarits_dir = self.models_dir / "gabarits"
        
        for dir_path in [self.cv_dir, self.nlp_dir, self.gabarits_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.loaded_models = {}
        self.model_checksums = {}
        
    def _calculate_checksum(self, file_path):
        """Calcule le checksum MD5 d'un fichier"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_model_info(self, model_name, model_type, checksum, metadata=None):
        """Sauvegarde les informations d'un modèle"""
        info_file = self.models_dir / f"{model_type}_info.json"
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
        else:
            info = {}
        
        info[model_name] = {
            "checksum": checksum,
            "metadata": metadata or {},
            "path": str(self.models_dir / model_type / model_name)
        }
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def download_cv_models(self):
        """Télécharge les modèles Computer Vision pré-entraînés"""
        logger.info("Téléchargement des modèles Computer Vision...")
        
        models_to_download = {
            "resnet50": models.resnet50,
            "efficientnet_b0": lambda: models.efficientnet_b0(weights='DEFAULT')
        }
        
        for model_name, model_fn in models_to_download.items():
            try:
                logger.info(f"Téléchargement de {model_name}...")
                model = model_fn(pretrained=True)
                
                # Sauvegarder le modèle
                model_path = self.cv_dir / f"{model_name}.pth"
                torch.save(model.state_dict(), model_path)
                
                # Calculer et sauvegarder le checksum
                checksum = self._calculate_checksum(model_path)
                self._save_model_info(
                    f"{model_name}.pth",
                    "cv",
                    checksum,
                    {"architecture": model_name}
                )
                
                logger.info(f"{model_name} téléchargé et sauvegardé avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement de {model_name}: {e}")
    
    def download_nlp_models(self):
        """Télécharge les modèles NLP pré-entraînés"""
        logger.info("Téléchargement des modèles NLP...")
        
        models_to_download = {
            "camembert-base": "camembert-base"
        }
        
        for model_name, model_id in models_to_download.items():
            try:
                logger.info(f"Téléchargement de {model_name}...")
                
                # Télécharger le modèle et le tokenizer
                model = AutoModel.from_pretrained(model_id)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Sauvegarder le modèle
                model_path = self.nlp_dir / model_name
                model_path.mkdir(exist_ok=True)
                model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                
                logger.info(f"{model_name} téléchargé et sauvegardé avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement de {model_name}: {e}")
    
    def load_cv_model(self, model_name="resnet50", device="cpu"):
        """Charge un modèle Computer Vision depuis le stockage local"""
        cache_key = f"cv_{model_name}_{device}"
        
        if cache_key in self.loaded_models:
            logger.info(f"Modèle {model_name} déjà chargé en cache")
            return self.loaded_models[cache_key]
        
        model_path = self.cv_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle {model_name} non trouvé. Exécutez setup_offline.py d'abord."
            )
        
        logger.info(f"Chargement du modèle {model_name}...")
        
        # Charger l'architecture appropriée
        if model_name == "resnet50":
            model = models.resnet50(pretrained=False)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
        else:
            raise ValueError(f"Architecture {model_name} non supportée")
        
        # Charger les poids
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Mettre en cache
        self.loaded_models[cache_key] = model
        
        return model
    
    def load_nlp_model(self, model_name="camembert-base", device="cpu"):
        """Charge un modèle NLP depuis le stockage local"""
        cache_key = f"nlp_{model_name}_{device}"
        
        if cache_key in self.loaded_models:
            logger.info(f"Modèle {model_name} déjà chargé en cache")
            return self.loaded_models[cache_key]
        
        model_path = self.nlp_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle {model_name} non trouvé. Exécutez setup_offline.py d'abord."
            )
        
        logger.info(f"Chargement du modèle {model_name}...")
        
        # Charger le modèle et le tokenizer
        model = AutoModel.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        model = model.to(device)
        model.eval()
        
        # Mettre en cache
        self.loaded_models[cache_key] = {
            "model": model,
            "tokenizer": tokenizer
        }
        
        return model, tokenizer
    
    def verify_model_integrity(self, model_name, model_type):
        """Vérifie l'intégrité d'un modèle sauvegardé"""
        info_file = self.models_dir / f"{model_type}_info.json"
        
        if not info_file.exists():
            logger.warning(f"Fichier d'information pour {model_type} non trouvé")
            return False
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        if model_name not in info:
            logger.warning(f"Modèle {model_name} non trouvé dans les informations")
            return False
        
        model_path = Path(info[model_name]["path"])
        
        if not model_path.exists():
            logger.error(f"Fichier modèle {model_path} non trouvé")
            return False
        
        # Vérifier le checksum
        current_checksum = self._calculate_checksum(model_path)
        saved_checksum = info[model_name]["checksum"]
        
        if current_checksum != saved_checksum:
            logger.error(f"Checksum ne correspond pas pour {model_name}")
            return False
        
        logger.info(f"Intégrité du modèle {model_name} vérifiée")
        return True
    
    def clear_cache(self):
        """Vide le cache des modèles chargés"""
        self.loaded_models.clear()
        logger.info("Cache des modèles vidé")
    
    def get_model_info(self, model_type):
        """Récupère les informations sur tous les modèles d'un type"""
        info_file = self.models_dir / f"{model_type}_info.json"
        
        if not info_file.exists():
            return {}
        
        with open(info_file, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    manager = OfflineModelManager()
    print("OfflineModelManager initialisé")

