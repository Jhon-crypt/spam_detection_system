"""
Feature extraction module for spam detection.
Implements TF-IDF vectorization and other text features.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Tuple
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction for text classification using TF-IDF and additional features.
    """
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 use_additional_features: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            use_additional_features: Whether to include additional text features
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_additional_features = use_additional_features
        
        # Initialize vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.is_fitted = False
        
    def _extract_additional_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract additional text features.
        
        Args:
            texts: List of text documents
            
        Returns:
            Array of additional features
        """
        features = []
        
        for text in texts:
            text_features = []
            
            # Text length features
            text_features.append(len(text))  # Character count
            text_features.append(len(text.split()))  # Word count
            
            # Punctuation features
            text_features.append(text.count('!'))  # Exclamation marks
            text_features.append(text.count('?'))  # Question marks
            text_features.append(text.count('$'))  # Dollar signs
            
            # Case features
            text_features.append(sum(1 for c in text if c.isupper()))  # Uppercase count
            text_features.append(len([word for word in text.split() if word.isupper()]))  # All caps words
            
            # Special patterns
            text_features.append(1 if 'free' in text.lower() else 0)  # Contains 'free'
            text_features.append(1 if 'click' in text.lower() else 0)  # Contains 'click'
            text_features.append(1 if 'urgent' in text.lower() else 0)  # Contains 'urgent'
            text_features.append(1 if 'winner' in text.lower() else 0)  # Contains 'winner'
            text_features.append(1 if 'money' in text.lower() else 0)  # Contains 'money'
            
            # URL and email patterns (simplified)
            text_features.append(text.lower().count('http'))  # URL count
            text_features.append(text.count('@'))  # Email count
            
            features.append(text_features)
            
        return np.array(features)
    
    def fit(self, X: List[str], y: Optional[List] = None):
        """
        Fit the feature extractor to the training data.
        
        Args:
            X: List of text documents
            y: Target labels (not used, for sklearn compatibility)
            
        Returns:
            self
        """
        logger.info("Fitting TF-IDF vectorizer...")
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(X)
        
        self.is_fitted = True
        logger.info(f"Feature extractor fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")
        
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            X: List of text documents
            
        Returns:
            Feature matrix
            
        Raises:
            ValueError: If not fitted
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
            
        # Get TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(X).toarray()
        
        if self.use_additional_features:
            # Get additional features
            additional_features = self._extract_additional_features(X)
            
            # Combine features
            features = np.hstack([tfidf_features, additional_features])
            
            logger.info(f"Extracted {features.shape[1]} total features "
                       f"({tfidf_features.shape[1]} TF-IDF + {additional_features.shape[1]} additional)")
        else:
            features = tfidf_features
            logger.info(f"Extracted {features.shape[1]} TF-IDF features")
            
        return features
    
    def fit_transform(self, X: List[str], y: Optional[List] = None) -> np.ndarray:
        """
        Fit the feature extractor and transform the data.
        
        Args:
            X: List of text documents
            y: Target labels (not used, for sklearn compatibility)
            
        Returns:
            Feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If not fitted
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")
            
        feature_names = list(self.tfidf_vectorizer.get_feature_names_out())
        
        if self.use_additional_features:
            additional_names = [
                'char_count', 'word_count', 'exclamation_count', 'question_count',
                'dollar_count', 'uppercase_count', 'caps_words_count', 'contains_free',
                'contains_click', 'contains_urgent', 'contains_winner', 'contains_money',
                'url_count', 'email_count'
            ]
            feature_names.extend(additional_names)
            
        return feature_names
    
    def get_top_features(self, n_features: int = 20) -> List[Tuple[str, float]]:
        """
        Get top features by average TF-IDF score.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            List of (feature_name, score) tuples
            
        Raises:
            ValueError: If not fitted
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting top features")
            
        # Get feature names and calculate average scores
        feature_names = list(self.tfidf_vectorizer.get_feature_names_out())
        
        # This is a simplified approach - in practice, you might want to use
        # the actual TF-IDF scores from your training data
        vocab = self.tfidf_vectorizer.vocabulary_
        idf_scores = self.tfidf_vectorizer.idf_
        
        # Create list of (feature, idf_score) tuples
        feature_scores = [(name, idf_scores[vocab[name]]) for name in feature_names]
        
        # Sort by IDF score (lower IDF = more common, higher discriminative power)
        feature_scores.sort(key=lambda x: x[1])
        
        return feature_scores[:n_features]
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted feature extractor.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If not fitted
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before saving")
            
        joblib.dump(self, filepath)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """
        Load a fitted feature extractor.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded feature extractor
        """
        extractor = joblib.load(filepath)
        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor