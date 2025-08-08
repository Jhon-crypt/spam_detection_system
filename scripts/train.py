"""
Training script for the spam detection system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.spam_detector import SpamClassifier
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Train a spam classifier model."""
    parser = argparse.ArgumentParser(description='Train Spam Classifier')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--model-type', choices=['naive_bayes', 'svm', 'random_forest', 'logistic'],
                       default='naive_bayes', help='Type of model to train')
    parser.add_argument('--output', default='models/spam_classifier.joblib', 
                       help='Output path for trained model')
    parser.add_argument('--grid-search', action='store_true',
                       help='Use grid search for hyperparameters')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize classifier
    classifier = SpamClassifier(
        model_type=args.model_type,
        use_preprocessing=True,
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    # Load data and train
    classifier.load_data(args.data)
    results = classifier.train(
        validation_split=0.2,
        use_grid_search=args.grid_search,
        cv_folds=5
    )
    
    # Save model
    classifier.save_model(args.output)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {args.output}")
    print(f"F1-Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")

if __name__ == '__main__':
    main()