"""
Module 4 : Traitement du Langage Naturel Offline
Classification textuelle avec OCR, motifs, CamemBERT et RNN
"""

from .nlp_classifier import NLPClassifier
from .ocr_extractor import OCRExtractor
from .semantic_patterns import SemanticPatternMatcher

__all__ = ['NLPClassifier', 'OCRExtractor', 'SemanticPatternMatcher']

