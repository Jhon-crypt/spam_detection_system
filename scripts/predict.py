"""
Prediction script for the spam detection system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.spam_detector import SpamClassifier
import argparse

def main():
    """Make predictions using trained model."""
    parser = argparse.ArgumentParser(description='Predict with Spam Classifier')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--text', nargs='+', required=True, help='Email text(s) to classify')
    
    args = parser.parse_args()
    
    # Load model
    classifier = SpamClassifier.load_model(args.model)
    
    # Make predictions
    results = classifier.predict(args.text)
    
    if isinstance(results, dict):
        results = [results]
    
    print("\nPrediction Results:")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nEmail {i}:")
        print(f"Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        print(f"Classification: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Spam Probability: {result['spam_probability']:.4f}")

if __name__ == '__main__':
    main()