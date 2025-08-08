"""
Spam Detection System

A machine learning-based email classifier that distinguishes between spam and legitimate emails.
"""

from .classifier import SpamClassifier
from .preprocessor import TextPreprocessor
from .features import FeatureExtractor
from .data import DataLoader
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__author__ = "Spam Detection Team"

__all__ = [
    'SpamClassifier',
    'TextPreprocessor', 
    'FeatureExtractor',
    'DataLoader',
    'ModelEvaluator'
]