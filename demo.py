"""
Demo script for the spam detection system.
"""

import os
import sys
from src.spam_detector import SpamClassifier, DataLoader

def main():
    """Run a complete demo of the spam detection system."""
    print("🚀 Spam Detection System - Demo")
    print("=" * 50)
    
    # Check for sample data
    data_path = 'data/sample_emails.csv'
    if not os.path.exists(data_path):
        print(f"❌ Sample data not found at {data_path}")
        return
    
    print("📊 Loading sample data...")
    
    # Initialize classifier
    classifier = SpamClassifier(
        model_type='naive_bayes',
        use_preprocessing=True,
        max_features=3000,
        ngram_range=(1, 2)
    )
    
    # Load and train
    classifier.load_data(data_path)
    print("🔥 Training classifier...")
    results = classifier.train(validation_split=0.3)
    
    print(f"\n✅ Training completed!")
    print(f"   F1-Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    print(f"   Features: {results['features_count']}")
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_emails = [
        "Congratulations! You've won $10,000! Click here to claim now!",
        "Hi, can we schedule a meeting for next Tuesday at 2pm?",
        "URGENT: Your account will be suspended! Verify immediately!",
        "Please find the quarterly report attached for your review.",
        "Free money! No strings attached! Act now!",
        "The team meeting has been moved to Conference Room B."
    ]
    
    for i, email in enumerate(test_emails, 1):
        result = classifier.predict(email)
        status = "🔴 SPAM" if result['label'] == 'spam' else "🟢 HAM"
        print(f"\n{i}. {status} (confidence: {result['confidence']:.3f})")
        print(f"   Text: {email[:60]}...")
    
    # Save model
    model_path = 'models/demo_model.joblib'
    os.makedirs('models', exist_ok=True)
    classifier.save_model(model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"\nNext steps:")
    print(f"• Train with your data: python scripts/train.py --data your_data.csv")
    print(f"• Make predictions: python scripts/predict.py --model {model_path} --text 'Your email here'")

if __name__ == '__main__':
    main()