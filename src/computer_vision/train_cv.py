"""
Script d'entraînement du modèle Computer Vision Hybride
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .hybrid_cv_model import HybridCVModel, LightHybridCVModel
import sys
# Ajouter le parent (src) au path pour les imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
from src.gabarits.detector_gabarits import DetectorGabarits
from src.preprocessing.pdf_to_image import PDFToImageConverter
from src.preprocessing.image_preprocessor import ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentDataset(Dataset):
    """Dataset pour les documents administratifs"""
    
    def __init__(
        self,
        data_dir: str,
        classes: list,
        transform=None,
        split: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.15
    ):
        """
        Args:
            data_dir: Dossier contenant les sous-dossiers par classe
            classes: Liste des noms de classes
            transform: Transformations à appliquer
            split: "train", "val" ou "test"
            train_split: Proportion pour l'entraînement
            val_split: Proportion pour la validation
        """
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = transform
        self.split = split
        
        # Charger les images et labels
        self.samples = []
        self._load_samples(train_split, val_split)
        
        # Initialiser le détecteur de gabarits
        self.gabarit_detector = DetectorGabarits()
        
        logger.info(f"Dataset {split} créé: {len(self.samples)} échantillons")
    
    def _load_samples(self, train_split: float, val_split: float):
        """Charge les échantillons avec split train/val/test"""
        all_samples = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Dossier {class_dir} non trouvé")
                continue
            
            # Lister les images
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for img_file in image_files:
                all_samples.append((str(img_file), self.class_to_idx[class_name]))
        
        # Mélanger
        np.random.seed(42)
        np.random.shuffle(all_samples)
        
        # Split
        n_total = len(all_samples)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        if self.split == "train":
            self.samples = all_samples[:n_train]
        elif self.split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:  # test
            self.samples = all_samples[n_train + n_val:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Charger l'image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        # Extraire les features de gabarits
        # Convertir PIL à numpy pour le détecteur
        img_np = np.array(Image.open(img_path).convert('RGB'))
        features = self.gabarit_detector.extract_all_features(img_np)
        
        # Sélectionner les features pertinentes (10 features principales)
        gabarit_features = np.array([
            features.get('aspect_ratio', 0.0),
            features.get('photo_detection', 0.0),
            features.get('photo_confidence', 0.0),
            features.get('tabular_structure', 0.0),
            features.get('tabular_score', 0.0),
            features.get('text_density', 0.0),
            features.get('numeric_density', 0.0),
            features.get('signature_zone', 0.0),
            features.get('signature_confidence', 0.0),
            features.get('vertical_alignment', 0.0)
        ], dtype=np.float32)
        
        return image, torch.from_numpy(gabarit_features), label


def train_hybrid_model(
    data_dir: str,
    classes: list,
    output_dir: str = "models/cv",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_light_model: bool = False
):
    """
    Entraîne le modèle hybride
    
    Args:
        data_dir: Dossier contenant les images par classe
        classes: Liste des noms de classes
        output_dir: Dossier de sortie pour sauvegarder le modèle
        epochs: Nombre d'époques
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
        device: Device (cuda ou cpu)
        use_light_model: Utiliser le modèle léger
    """
    logger.info(f"Entraînement sur device: {device}")
    
    # Transformations avec augmentation de données
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.RandomAdjustSharpness(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = DocumentDataset(data_dir, classes, train_transform, split="train")
    val_dataset = DocumentDataset(data_dir, classes, val_transform, split="val")
    
    # DataLoaders (optimisé pour CPU: moins de workers)
    num_workers = 2 if device == "cpu" else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Modèle
    if use_light_model:
        model = LightHybridCVModel(num_classes=len(classes), gabarit_features_dim=10)
    else:
        model = HybridCVModel(num_classes=len(classes), gabarit_features_dim=10)
    
    model = model.to(device)
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Début de l'entraînement...")
    
    for epoch in range(epochs):
        # Entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, gabarit_features, labels in pbar:
            images = images.to(device)
            gabarit_features = gabarit_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, gabarit_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * train_correct / train_total})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, gabarit_features, labels in val_loader:
                images = images.to(device)
                gabarit_features = gabarit_features.to(device)
                labels = labels.to(device)
                
                outputs = model(images, gabarit_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Mise à jour du scheduler
        scheduler.step(val_loss)
        
        # Historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'classes': classes
            }, output_path / "best_model.pth")
            logger.info(f"Meilleur modèle sauvegardé (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping après {epoch+1} époques")
            break
    
    # Sauvegarder l'historique
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Évaluation finale
    logger.info("Évaluation sur le jeu de validation...")
    evaluate_model(model, val_loader, device, classes, output_path)
    
    # Tracer les courbes
    plot_training_history(history, output_path)
    
    logger.info("Entraînement terminé!")


def evaluate_model(model, data_loader, device, classes, output_dir):
    """Évalue le modèle et génère un rapport"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, gabarit_features, labels in tqdm(data_loader, desc="Évaluation"):
            images = images.to(device)
            gabarit_features = gabarit_features.to(device)
            labels = labels.to(device)
            
            outputs = model(images, gabarit_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métriques
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes)
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info(f"\nAccuracy globale: {accuracy:.4f}")
    logger.info(f"\nRapport de classification:\n{report}")
    
    # Sauvegarder le rapport
    with open(Path(output_dir) / "evaluation_report.txt", 'w') as f:
        f.write(f"Accuracy globale: {accuracy:.4f}\n\n")
        f.write("Rapport de classification:\n")
        f.write(report)
        f.write("\n\nMatrice de confusion:\n")
        f.write(str(cm))
    
    # Visualiser la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "confusion_matrix.png")
    plt.close()


def plot_training_history(history, output_dir):
    """Trace les courbes d'entraînement"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_history.png")
    plt.close()


if __name__ == "__main__":
    # Configuration
    data_dir = "data/preprocessed_images"
    classes = ["identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"]
    
    train_hybrid_model(
        data_dir=data_dir,
        classes=classes,
        epochs=50,
        batch_size=32,
        learning_rate=0.0001
    )

