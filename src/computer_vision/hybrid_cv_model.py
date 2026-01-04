"""
Modèle Computer Vision Hybride
Combine CNN (ResNet50) avec features de gabarits
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridCVModel(nn.Module):
    """
    Modèle hybride combinant CNN et features de gabarits
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "resnet50",
        gabarit_features_dim: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.5
    ):
        """
        Initialise le modèle hybride
        
        Args:
            num_classes: Nombre de classes (5 par défaut)
            backbone_name: Nom du backbone CNN ("resnet50" ou "efficientnet_b0")
            gabarit_features_dim: Dimension des features de gabarits
            hidden_dim: Dimension de la couche cachée pour les gabarits
            dropout: Taux de dropout
        """
        super(HybridCVModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.gabarit_features_dim = gabarit_features_dim
        
        # Branche CNN
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            # Remplacer la dernière couche
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # On va utiliser Global Average Pooling
            self.cnn_features_dim = 2048
        elif backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.cnn_features_dim = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} non supporté")
        
        # Global Average Pooling pour CNN
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Branche Gabarits
        self.gabarit_branch = nn.Sequential(
            nn.Linear(gabarit_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion des features
        fusion_dim = self.cnn_features_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classificateur final
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, image: torch.Tensor, gabarit_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image: Tensor d'image [batch_size, 3, H, W]
            gabarit_features: Tensor de features gabarits [batch_size, gabarit_features_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Branche CNN
        cnn_features = self.backbone.conv1(image)
        cnn_features = self.backbone.bn1(cnn_features)
        cnn_features = self.backbone.relu(cnn_features)
        cnn_features = self.backbone.maxpool(cnn_features)
        
        cnn_features = self.backbone.layer1(cnn_features)
        cnn_features = self.backbone.layer2(cnn_features)
        cnn_features = self.backbone.layer3(cnn_features)
        cnn_features = self.backbone.layer4(cnn_features)
        
        # Global Average Pooling
        cnn_features = self.global_pool(cnn_features)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Branche Gabarits
        gabarit_embeddings = self.gabarit_branch(gabarit_features)
        
        # Fusion
        combined_features = torch.cat([cnn_features, gabarit_embeddings], dim=1)
        fused_features = self.fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict(self, image: torch.Tensor, gabarit_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédiction avec probabilités
        
        Args:
            image: Tensor d'image [batch_size, 3, H, W]
            gabarit_features: Tensor de features gabarits [batch_size, gabarit_features_dim]
            
        Returns:
            Tuple (probabilités, classes prédites)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(image, gabarit_features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return probs, preds


class LightHybridCVModel(nn.Module):
    """
    Version légère du modèle hybride pour machines peu puissantes
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        gabarit_features_dim: int = 10,
        hidden_dim: int = 32,
        dropout: float = 0.3
    ):
        super(LightHybridCVModel, self).__init__()
        
        # CNN léger (MobileNet)
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.cnn_features_dim = 1280
        
        # Branche Gabarits (plus petite)
        self.gabarit_branch = nn.Sequential(
            nn.Linear(gabarit_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion
        fusion_dim = self.cnn_features_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classificateur
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, image: torch.Tensor, gabarit_features: torch.Tensor) -> torch.Tensor:
        # CNN
        cnn_features = self.backbone.features(image)
        cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1))
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Gabarits
        gabarit_embeddings = self.gabarit_branch(gabarit_features)
        
        # Fusion
        combined = torch.cat([cnn_features, gabarit_embeddings], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


if __name__ == "__main__":
    # Test du modèle
    model = HybridCVModel(num_classes=5, gabarit_features_dim=10)
    print(f"Modèle créé avec succès")
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_gabarits = torch.randn(batch_size, 10)
    
    output = model(dummy_image, dummy_gabarits)
    print(f"Output shape: {output.shape}")

