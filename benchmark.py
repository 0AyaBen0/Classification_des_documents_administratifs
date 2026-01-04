"""
Script de Benchmarking
Compare les temps de traitement et consommation des ressources de chaque modèle
"""

import sys
import time
import psutil
import torch
from pathlib import Path
import json
from datetime import datetime

# Ajouter le dossier src au path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.utils.logger_setup import setup_logger
from src.offline_manager import OfflineModelManager
from src.gabarits.detector_gabarits import DetectorGabarits
from src.nlp.nlp_classifier import NLPClassifier
from src.preprocessing.pdf_to_image import PDFToImageConverter
from src.preprocessing.image_preprocessor import ImagePreprocessor
import cv2
import numpy as np

logger = setup_logger()


class Benchmark:
    """Classe pour benchmarker les différents composants"""
    
    def __init__(self):
        self.results = []
    
    def measure_memory(self):
        """Mesure la mémoire utilisée"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def benchmark_cv_model(self, model_name="resnet50", device="cpu", num_runs=10):
        """Benchmark un modèle CV"""
        logger.info(f"Benchmark CV: {model_name} sur {device}")
        
        manager = OfflineModelManager()
        
        # Mesurer le chargement
        start_time = time.time()
        model = manager.load_cv_model(model_name, device)
        load_time = time.time() - start_time
        load_memory = self.measure_memory()
        
        # Préparer une image de test
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        dummy_gabarits = torch.randn(1, 10).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(dummy_image, dummy_gabarits)
        
        # Mesurer l'inférence
        inference_times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_image, dummy_gabarits)
            inference_times.append(time.time() - start)
        
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        inference_memory = self.measure_memory() - load_memory
        
        result = {
            "component": "CV",
            "model": model_name,
            "device": device,
            "load_time": load_time,
            "load_memory_mb": load_memory,
            "avg_inference_time": avg_inference,
            "std_inference_time": std_inference,
            "inference_memory_mb": inference_memory,
            "num_runs": num_runs
        }
        
        self.results.append(result)
        logger.info(f"  Temps chargement: {load_time:.3f}s")
        logger.info(f"  Temps inférence moyen: {avg_inference:.3f}s ± {std_inference:.3f}s")
        logger.info(f"  Mémoire: {load_memory:.1f} MB")
        
        return result
    
    def benchmark_nlp_model(self, device="cpu", num_runs=10):
        """Benchmark le modèle NLP"""
        logger.info(f"Benchmark NLP: CamemBERT sur {device}")
        
        manager = OfflineModelManager()
        
        # Mesurer le chargement
        start_time = time.time()
        model, tokenizer = manager.load_nlp_model("camembert-base", device)
        load_time = time.time() - start_time
        load_memory = self.measure_memory()
        
        # Texte de test
        test_text = "Ceci est un document administratif contenant des informations importantes."
        
        # Warmup
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        
        # Mesurer l'inférence
        inference_times = []
        for _ in range(num_runs):
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            start = time.time()
            with torch.no_grad():
                _ = model(**inputs)
            inference_times.append(time.time() - start)
        
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        inference_memory = self.measure_memory() - load_memory
        
        result = {
            "component": "NLP",
            "model": "camembert-base",
            "device": device,
            "load_time": load_time,
            "load_memory_mb": load_memory,
            "avg_inference_time": avg_inference,
            "std_inference_time": std_inference,
            "inference_memory_mb": inference_memory,
            "num_runs": num_runs
        }
        
        self.results.append(result)
        logger.info(f"  Temps chargement: {load_time:.3f}s")
        logger.info(f"  Temps inférence moyen: {avg_inference:.3f}s ± {std_inference:.3f}s")
        logger.info(f"  Mémoire: {load_memory:.1f} MB")
        
        return result
    
    def benchmark_gabarits(self, num_runs=10):
        """Benchmark le détecteur de gabarits"""
        logger.info("Benchmark Gabarits")
        
        detector = DetectorGabarits()
        
        # Image de test
        test_image = np.random.randint(0, 255, (1000, 700, 3), dtype=np.uint8)
        
        # Warmup
        _ = detector.extract_all_features(test_image)
        
        # Mesurer
        inference_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = detector.extract_all_features(test_image)
            inference_times.append(time.time() - start)
        
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        
        result = {
            "component": "Gabarits",
            "model": "DetectorGabarits",
            "device": "cpu",
            "avg_inference_time": avg_inference,
            "std_inference_time": std_inference,
            "num_runs": num_runs
        }
        
        self.results.append(result)
        logger.info(f"  Temps moyen: {avg_inference:.3f}s ± {std_inference:.3f}s")
        
        return result
    
    def benchmark_ocr(self, num_runs=5):
        """Benchmark OCR"""
        logger.info("Benchmark OCR")
        
        from src.nlp.ocr_extractor import OCRExtractor
        extractor = OCRExtractor()
        
        # Image de test avec texte
        test_image = np.ones((1000, 700, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Test Document", (50, 500), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Mesurer
        inference_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = extractor.extract_text(test_image)
            inference_times.append(time.time() - start)
        
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        
        result = {
            "component": "OCR",
            "model": "Tesseract",
            "device": "cpu",
            "avg_inference_time": avg_inference,
            "std_inference_time": std_inference,
            "num_runs": num_runs
        }
        
        self.results.append(result)
        logger.info(f"  Temps moyen: {avg_inference:.3f}s ± {std_inference:.3f}s")
        
        return result
    
    def save_results(self, output_path="benchmark_results.json"):
        """Sauvegarde les résultats"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "cuda_available": torch.cuda.is_available()
            },
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Résultats sauvegardés: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark des modèles")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--runs", type=int, default=10)
    
    args = parser.parse_args()
    
    benchmark = Benchmark()
    
    logger.info("=" * 60)
    logger.info("Début du benchmarking")
    logger.info("=" * 60)
    
    # Benchmark CV
    benchmark.benchmark_cv_model("resnet50", args.device, args.runs)
    
    # Benchmark NLP
    benchmark.benchmark_nlp_model(args.device, args.runs)
    
    # Benchmark Gabarits
    benchmark.benchmark_gabarits(args.runs)
    
    # Benchmark OCR
    benchmark.benchmark_ocr(min(args.runs, 5))  # OCR est plus lent
    
    logger.info("=" * 60)
    logger.info("Benchmark terminé")
    logger.info("=" * 60)
    
    # Sauvegarder
    benchmark.save_results(args.output)
    
    # Afficher un résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for result in benchmark.results:
        print(f"{result['component']} ({result.get('model', 'N/A')}): "
              f"{result.get('avg_inference_time', 0):.3f}s")


if __name__ == "__main__":
    main()

