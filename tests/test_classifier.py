"""
Tests for the spam classifier.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.spam_detector import SpamClassifier, DataLoader

class TestSpamClassifier:
    """Test cases for SpamClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = SpamClassifier()
        assert classifier.model_type == 'naive_bayes'
        assert classifier.use_preprocessing == True
        assert not classifier.is_trained
    
    def test_different_models(self):
        """Test different model types."""
        models = ['naive_bayes', 'svm', 'random_forest', 'logistic']
        
        for model_type in models:
            classifier = SpamClassifier(model_type=model_type)
            assert classifier.model_type == model_type
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        classifier = SpamClassifier()
        
        with pytest.raises(ValueError):
            classifier.predict("This is a test email")
    
    def test_training_with_sample_data(self):
        """Test training with sample data."""
        # Create minimal test data
        texts = [
            "Free money! Click now!",
            "Meeting at 2pm tomorrow",
            "Win big! Act fast!",
            "Project deadline extended"
        ]
        labels = [1, 0, 1, 0]  # spam, ham, spam, ham
        
        classifier = SpamClassifier()
        classifier.data_loader.load_from_lists(texts, labels)
        
        results = classifier.train(validation_split=0.5)
        
        assert classifier.is_trained
        assert 'cv_mean' in results
        assert results['training_samples'] == 4
    
    def test_prediction_after_training(self):
        """Test prediction after training."""
        # Create test data
        texts = [
            "Free money! Click now!",
            "Meeting at 2pm tomorrow", 
            "Win lottery! Act fast!",
            "Project deadline extended",
            "Urgent! Verify account!",
            "Team lunch on Friday"
        ]
        labels = [1, 0, 1, 0, 1, 0]
        
        classifier = SpamClassifier()
        classifier.data_loader.load_from_lists(texts, labels)
        classifier.train()
        
        # Test prediction
        result = classifier.predict("Free cash! Click here!")
        
        assert 'label' in result
        assert 'confidence' in result
        assert result['label'] in ['spam', 'ham']
        assert 0 <= result['confidence'] <= 1
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        texts = ["Free money!", "Meeting today", "Win now!", "Project update"]
        labels = [1, 0, 1, 0]
        
        classifier = SpamClassifier()
        classifier.data_loader.load_from_lists(texts, labels)
        classifier.train()
        
        test_emails = ["Free cash!", "Team meeting"]
        results = classifier.predict(test_emails)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all('label' in result for result in results)

if __name__ == '__main__':
    pytest.main([__file__])