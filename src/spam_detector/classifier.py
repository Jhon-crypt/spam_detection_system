"""
Main spam classification module.
Integrates all components for training and predicting spam emails.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Tuple, Union
import joblib
import logging
import os

from .preprocessor import TextPreprocessor
from .features import FeatureExtractor
from .data import DataLoader
from .evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpamClassifier:
    """
    Complete spam email classifier with preprocessing, feature extraction,
    training, and evaluation capabilities.
    """
    
    def __init__(self, model_type: str = 'naive_bayes', 
                 use_preprocessing: bool = True,
                 **feature_params):
        """
        Initialize the spam classifier.
        
        Args:
            model_type: Type of classifier ('naive_bayes', 'svm', 'random_forest', 'logistic')
            use_preprocessing: Whether to use text preprocessing
            **feature_params: Parameters for feature extraction
        """
        self.model_type = model_type
        self.use_preprocessing = use_preprocessing
        
        # Initialize components
        if use_preprocessing:
            self.preprocessor = TextPreprocessor()
        else:
            self.preprocessor = None
            
        self.feature_extractor = FeatureExtractor(**feature_params)
        self.data_loader = DataLoader()
        self.evaluator = ModelEvaluator()
        
        # Initialize model
        self.model = self._get_model(model_type)
        self.is_trained = False
        
        # Store training history
        self.training_history = {}
        
    def _get_model(self, model_type: str):
        """
        Get the specified model.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Initialized model
            
        Raises:
            ValueError: If model type is not supported
        """
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Choose from: {list(models.keys())}")
            
        return models[model_type]
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts if preprocessing is enabled.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            Preprocessed texts
        """
        if self.preprocessor:
            return self.preprocessor.preprocess_batch(texts)
        return texts
    
    def load_data(self, file_path: str, text_column: str = 'text', 
                  label_column: str = 'label') -> None:
        """
        Load training data from file.
        
        Args:
            file_path: Path to the data file
            text_column: Name of text column
            label_column: Name of label column
        """
        self.data_loader.load_csv(file_path, text_column, label_column)
        logger.info("Data loaded successfully")
        
        # Print data info
        info = self.data_loader.get_data_info()
        for key, value in info.items():
            logger.info(f"{key}: {value}")
    
    def train(self, X_train: Optional[List[str]] = None, 
              y_train: Optional[List] = None,
              validation_split: float = 0.2,
              use_grid_search: bool = False,
              cv_folds: int = 5) -> Dict:
        """
        Train the spam classifier.
        
        Args:
            X_train: Training texts (if None, uses loaded data)
            y_train: Training labels (if None, uses loaded data)
            validation_split: Proportion for validation
            use_grid_search: Whether to use grid search for hyperparameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {self.model_type} classifier...")
        
        # Use provided data or split loaded data
        if X_train is None or y_train is None:
            if self.data_loader.X is None:
                raise ValueError("No data provided. Load data first or provide X_train and y_train")
            X_train, X_test, y_train, y_test = self.data_loader.split_data(
                test_size=validation_split, random_state=42)
        else:
            X_test, y_test = None, None
        
        # Preprocess texts
        X_train_processed = self._preprocess_texts(X_train.tolist() if hasattr(X_train, 'tolist') else X_train)
        
        # Extract features
        logger.info("Extracting features...")
        X_train_features = self.feature_extractor.fit_transform(X_train_processed)
        
        # Train model
        if use_grid_search:
            logger.info("Performing grid search for hyperparameters...")
            self.model = self._grid_search(X_train_features, y_train, cv_folds)
        else:
            self.model.fit(X_train_features, y_train)
        
        self.is_trained = True
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_features, y_train, 
                                   cv=cv_folds, scoring='f1')
        
        results = {
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'features_count': X_train_features.shape[1]
        }
        
        # Evaluate on validation set if available
        if X_test is not None and y_test is not None:
            val_results = self.evaluate(X_test.tolist(), y_test.tolist())
            results['validation_metrics'] = val_results['metrics']
        
        self.training_history = results
        
        logger.info(f"Training completed. CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def _grid_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                     cv_folds: int) -> object:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds
            
        Returns:
            Best model from grid search
        """
        param_grids = {
            'naive_bayes': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]},
            'svm': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']},
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            },
            'logistic': {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv_folds, 
                scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            # No grid search parameters defined, use default model
            self.model.fit(X_train, y_train)
            return self.model
    
    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict spam/ham for given texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Prediction results
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle single text
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]
        
        # Preprocess and extract features
        processed_texts = self._preprocess_texts(texts)
        features = self.feature_extractor.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i],
                'label': 'spam' if pred == 1 else 'ham',
                'confidence': max(prob),
                'spam_probability': prob[1] if len(prob) > 1 else prob[0],
                'ham_probability': prob[0] if len(prob) > 1 else 1 - prob[0]
            }
            results.append(result)
        
        return results[0] if single_text else results
    
    def evaluate(self, X_test: List[str], y_test: List, 
                 save_plots: bool = False, 
                 plots_dir: str = 'evaluation_plots') -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            save_plots: Whether to save evaluation plots
            plots_dir: Directory to save plots
            
        Returns:
            Evaluation results
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Preprocess and extract features
        X_test_processed = self._preprocess_texts(X_test)
        X_test_features = self.feature_extractor.transform(X_test_processed)
        
        # Make predictions
        y_pred = self.model.predict(X_test_features)
        y_proba = self.model.predict_proba(X_test_features)
        
        # Create evaluation directory
        if save_plots and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Get feature importance if available
        feature_names = self.feature_extractor.get_feature_names()
        feature_importance = None
        
        if hasattr(self.model, 'coef_'):
            # Linear models
            feature_importance = self.model.coef_[0]
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            feature_importance = self.model.feature_importances_
        
        # Evaluate
        results = self.evaluator.create_evaluation_report(
            y_test, y_pred, y_proba,
            feature_names=feature_names,
            feature_importance=feature_importance,
            save_dir=plots_dir if save_plots else None
        )
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and components.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SpamClassifier':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded SpamClassifier instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        classifier = cls.__new__(cls)
        classifier.model = model_data['model']
        classifier.feature_extractor = model_data['feature_extractor']
        classifier.preprocessor = model_data['preprocessor']
        classifier.model_type = model_data['model_type']
        classifier.training_history = model_data.get('training_history', {})
        classifier.is_trained = True
        
        # Initialize other components
        classifier.data_loader = DataLoader()
        classifier.evaluator = ModelEvaluator()
        classifier.use_preprocessing = classifier.preprocessor is not None
        
        logger.info(f"Model loaded from {filepath}")
        return classifier
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'use_preprocessing': self.use_preprocessing,
            'training_history': self.training_history
        }
        
        if self.is_trained and hasattr(self.feature_extractor, 'tfidf_vectorizer'):
            info['vocabulary_size'] = len(self.feature_extractor.tfidf_vectorizer.vocabulary_)
            info['feature_count'] = len(self.feature_extractor.get_feature_names())
        
        return info