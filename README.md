# 🚀 Spam Email Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)

A professional, production-ready machine learning system for classifying emails as spam or legitimate. Built with Python, scikit-learn, and NLTK, featuring multiple ML algorithms, comprehensive text preprocessing, and robust evaluation metrics.

## ✨ Key Features

- **🧠 Multiple ML Algorithms**: Naive Bayes, SVM, Random Forest, Logistic Regression
- **🔧 Advanced Text Processing**: Tokenization, stemming, stopword removal, feature engineering
- **📊 Smart Feature Extraction**: TF-IDF vectorization + custom text features (length, punctuation, keywords)
- **📈 Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC curves, confusion matrices
- **🏗️ Professional Architecture**: Modular design, proper packaging, comprehensive testing
- **⚡ Easy to Use**: Simple API, command-line tools, and interactive demo

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to project
git clone <your-repo-url>
cd spam_detection_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo.py
```

This will:
- Load sample email data
- Train a Naive Bayes classifier
- Evaluate performance with metrics
- Test predictions on sample emails
- Save the trained model

### 3. Make Predictions
```bash
# Using Python API
python -c "
from src.spam_detector import SpamClassifier
classifier = SpamClassifier.load_model('models/demo_model.joblib')
result = classifier.predict('Free money! Click here now!')
print(f'Classification: {result[\"label\"]} (confidence: {result[\"confidence\"]:.3f})')
"

# Using command-line scripts
python scripts/predict.py --model models/demo_model.joblib --text "Your email text here"
```

## 📖 Usage Examples

### Basic Classification
```python
from src.spam_detector import SpamClassifier

# Initialize classifier
classifier = SpamClassifier(model_type='naive_bayes')

# Load and train on your data
classifier.load_data('your_data.csv')  # CSV with 'text' and 'label' columns
results = classifier.train(validation_split=0.2)

# Make predictions
result = classifier.predict("Congratulations! You've won $1000!")
print(f"Spam probability: {result['spam_probability']:.3f}")

# Save model
classifier.save_model('my_spam_detector.joblib')
```

### Advanced Training
```python
# Train with different algorithms and hyperparameter tuning
classifier = SpamClassifier(
    model_type='random_forest',
    max_features=5000,
    ngram_range=(1, 2),
    use_additional_features=True
)

results = classifier.train(
    validation_split=0.2,
    use_grid_search=True,  # Automatic hyperparameter tuning
    cv_folds=5
)

# Comprehensive evaluation
eval_results = classifier.evaluate(
    test_texts, test_labels,
    save_plots=True,
    plots_dir='evaluation_results'
)
```

### Command Line Tools
```bash
# Train a model
python scripts/train.py \
    --data data/emails.csv \
    --model-type svm \
    --output models/svm_classifier.joblib \
    --grid-search

# Make predictions
python scripts/predict.py \
    --model models/svm_classifier.joblib \
    --text "URGENT: Verify your account now!" \
    --text "Meeting at 3pm in conference room"
```

## 🏗️ Project Structure

```
spam_detection_system/
├── src/spam_detector/          # Main package
│   ├── __init__.py            # Package exports
│   ├── classifier.py          # Main SpamClassifier class
│   ├── preprocessor.py        # Text preprocessing
│   ├── features.py            # Feature extraction
│   ├── data.py               # Data loading utilities
│   └── evaluation.py         # Model evaluation
├── scripts/                   # Command-line tools
│   ├── train.py              # Model training script
│   └── predict.py            # Prediction script
├── tests/                     # Unit tests
│   └── test_classifier.py    # Classifier tests
├── config/                    # Configuration files
│   └── model_config.py       # Model parameters
├── data/                      # Data files
│   └── sample_emails.csv     # Sample dataset
├── models/                    # Saved models
├── demo.py                   # Interactive demo
├── setup.py                  # Package installation
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🎯 Performance

The system achieves strong performance on email classification:

- **Accuracy**: ~90%+ on balanced datasets
- **Precision**: High precision for spam detection (low false positives)
- **Recall**: Good recall for legitimate emails (low false negatives)
- **Speed**: Fast training and prediction (<1s for typical datasets)

Performance varies based on:
- Dataset quality and size
- Feature engineering settings
- Model algorithm choice
- Hyperparameter tuning

## 🔧 Customization

### Model Configuration
```python
# Fast training (lower accuracy)
classifier = SpamClassifier(
    model_type='naive_bayes',
    max_features=2000,
    ngram_range=(1, 1),
    use_additional_features=False
)

# High accuracy (slower training)
classifier = SpamClassifier(
    model_type='random_forest',
    max_features=10000,
    ngram_range=(1, 3),
    use_additional_features=True
)
```

### Custom Features
The system extracts various text features:
- **TF-IDF vectors**: Word importance scores
- **Text statistics**: Length, word count, punctuation
- **Spam indicators**: Keywords like "free", "urgent", "click"
- **Format features**: URLs, emails, capitalization

### Data Format
Input CSV should have columns:
- `text`: Email content
- `label`: 'spam' or 'ham' (or 1/0)

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test with your own data
python demo.py  # Uses sample data
python scripts/train.py --data your_data.csv
```

## 📊 Model Comparison

| Algorithm | Speed | Accuracy | Memory | Best For |
|-----------|-------|----------|---------|----------|
| Naive Bayes | ⚡⚡⚡ | ⭐⭐⭐ | ⚡⚡⚡ | Quick prototyping |
| SVM | ⚡⚡ | ⭐⭐⭐⭐ | ⚡⚡ | High accuracy |
| Random Forest | ⚡ | ⭐⭐⭐⭐ | ⚡ | Robust performance |
| Logistic Regression | ⚡⚡ | ⭐⭐⭐ | ⚡⚡ | Interpretability |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Text processing powered by [NLTK](https://www.nltk.org/)
- Visualization with [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/spam-detection-system/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/spam-detection-system/wiki)

---

**Made with ❤️ for email security**