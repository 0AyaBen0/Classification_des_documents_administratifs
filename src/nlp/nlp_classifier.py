"""
Classificateur NLP complet
Combine motifs sémantiques, CamemBERT et RNN
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

from .ocr_extractor import OCRExtractor
from .semantic_patterns import SemanticPatternMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CamemBERTClassifier:
    """Classificateur basé sur CamemBERT"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialise le classificateur CamemBERT
        
        Args:
            model_path: Chemin vers le modèle (optionnel, utilise le modèle offline)
            device: Device (cpu ou cuda)
        """
        self.device = device
        
        if model_path is None:
            # Utiliser le modèle offline
            from ..offline_manager import OfflineModelManager
            manager = OfflineModelManager()
            self.model, self.tokenizer = manager.load_nlp_model("camembert-base", device)
        else:
            self.model = AutoModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = self.model.to(device)
        
        self.model.eval()
        self.max_length = 512
    
    def extract_embeddings(self, text: str) -> np.ndarray:
        """
        Extrait les embeddings [CLS] du texte
        
        Args:
            text: Texte à traiter
            
        Returns:
            Embedding vector
        """
        # Tokeniser
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Utiliser le token [CLS]
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def classify(self, text: str, num_classes: int = 5) -> np.ndarray:
        """
        Classification (nécessite un classificateur fine-tuné)
        Pour l'instant, retourne juste les embeddings
        
        Args:
            text: Texte à classifier
            num_classes: Nombre de classes
            
        Returns:
            Probabilités (placeholder)
        """
        # En production, il faudrait un classificateur fine-tuné
        # Pour l'instant, on retourne des probabilités uniformes
        embeddings = self.extract_embeddings(text)
        
        # Placeholder: probabilités uniformes
        # En production, utiliser un classificateur entraîné
        probs = np.ones(num_classes) / num_classes
        
        return probs


class RNNClassifier(nn.Module):
    """Classificateur RNN avec embeddings"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        """
        Initialise le classificateur RNN
        
        Args:
            vocab_size: Taille du vocabulaire
            embedding_dim: Dimension des embeddings
            hidden_dim: Dimension cachée
            num_layers: Nombre de couches
            num_classes: Nombre de classes
            dropout: Taux de dropout
        """
        super(RNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # Bidirectional
            num_heads=4,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_length]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Embeddings
        embedded = self.embedding(x)
        
        # RNN
        rnn_out, _ = self.rnn(embedded)
        
        # Attention
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        
        # Pooling (moyenne)
        pooled = attn_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class NLPClassifier:
    """Classificateur NLP complet combinant toutes les approches"""
    
    def __init__(
        self,
        use_camembert: bool = True,
        use_rnn: bool = False,
        use_patterns: bool = True,
        device: str = "cpu"
    ):
        """
        Initialise le classificateur NLP
        
        Args:
            use_camembert: Utiliser CamemBERT
            use_rnn: Utiliser RNN (nécessite entraînement)
            use_patterns: Utiliser les motifs sémantiques
            device: Device (cpu ou cuda)
        """
        self.device = device
        self.use_camembert = use_camembert
        self.use_rnn = use_rnn
        self.use_patterns = use_patterns
        
        # OCR
        self.ocr_extractor = OCRExtractor()
        
        # Motifs sémantiques
        if use_patterns:
            self.pattern_matcher = SemanticPatternMatcher()
        
        # CamemBERT
        if use_camembert:
            try:
                self.camembert = CamemBERTClassifier(device=device)
            except Exception as e:
                logger.warning(f"Impossible de charger CamemBERT: {e}")
                self.use_camembert = False
        
        # RNN (nécessite un modèle entraîné)
        if use_rnn:
            logger.warning("RNN nécessite un modèle entraîné. Non implémenté pour l'instant.")
            self.use_rnn = False
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extrait le texte d'une image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Texte extrait
        """
        return self.ocr_extractor.extract_text(image)
    
    def classify(self, text: str, classes: List[str]) -> Dict[str, float]:
        """
        Classifie un texte avec toutes les méthodes disponibles
        
        Args:
            text: Texte à classifier
            classes: Liste des noms de classes
            
        Returns:
            Dictionnaire {classe: probabilité}
        """
        results = {}
        
        # 1. Motifs sémantiques
        if self.use_patterns:
            pattern_scores = self.pattern_matcher.match_patterns(text)
            results['patterns'] = pattern_scores
        
        # 2. CamemBERT (placeholder pour l'instant)
        if self.use_camembert:
            try:
                camembert_probs = self.camembert.classify(text, len(classes))
                # Convertir en dictionnaire
                camembert_scores = {classes[i]: float(camembert_probs[i]) for i in range(len(classes))}
                results['camembert'] = camembert_scores
            except Exception as e:
                logger.warning(f"Erreur CamemBERT: {e}")
        
        # 3. Fusion des résultats
        final_scores = self._fuse_results(results, classes)
        
        return final_scores
    
    def _fuse_results(self, results: Dict, classes: List[str]) -> Dict[str, float]:
        """
        Fusionne les résultats de différentes méthodes
        
        Args:
            results: Résultats de chaque méthode
            classes: Liste des classes
            
        Returns:
            Scores finaux fusionnés
        """
        final_scores = {cls: 0.0 for cls in classes}
        total_weight = 0.0
        
        # Poids pour chaque méthode
        weights = {
            'patterns': 0.5,
            'camembert': 0.5,
            'rnn': 0.3
        }
        
        # Combiner les scores
        for method, scores in results.items():
            if method in weights and scores:
                weight = weights[method]
                total_weight += weight
                
                for cls in classes:
                    if cls in scores:
                        final_scores[cls] += scores[cls] * weight
        
        # Normaliser
        if total_weight > 0:
            for cls in classes:
                final_scores[cls] /= total_weight
        
        return final_scores
    
    def classify_image(self, image: np.ndarray, classes: List[str]) -> Tuple[str, float, Dict]:
        """
        Classifie une image (extraction OCR + classification)
        
        Args:
            image: Image en numpy array
            classes: Liste des noms de classes
            
        Returns:
            Tuple (classe_prédite, confiance, scores_détaillés)
        """
        # Extraction OCR
        text = self.extract_text(image)
        
        if not text or len(text.strip()) < 10:
            logger.warning("Texte extrait trop court ou vide")
            return None, 0.0, {}
        
        # Classification
        scores = self.classify(text, classes)
        
        if not scores:
            return None, 0.0, {}
        
        # Trouver la classe prédite
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        return predicted_class, confidence, scores


if __name__ == "__main__":
    classifier = NLPClassifier()
    print("NLPClassifier initialisé")

