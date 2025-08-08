"""
Configuration settings for the spam detection models.
"""

# Model configurations
MODEL_CONFIGS = {
    'naive_bayes': {
        'alpha': 1.0,
        'description': 'Fast and effective for text classification'
    },
    'svm': {
        'kernel': 'linear',
        'probability': True,
        'random_state': 42,
        'description': 'Good performance with proper tuning'
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'description': 'Robust ensemble method'
    },
    'logistic': {
        'random_state': 42,
        'max_iter': 1000,
        'description': 'Interpretable linear model'
    }
}

# Feature extraction settings
FEATURE_CONFIGS = {
    'default': {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'use_additional_features': True
    },
    'fast': {
        'max_features': 2000,
        'ngram_range': (1, 1),
        'min_df': 2,
        'max_df': 0.95,
        'use_additional_features': False
    },
    'comprehensive': {
        'max_features': 10000,
        'ngram_range': (1, 3),
        'min_df': 1,
        'max_df': 0.98,
        'use_additional_features': True
    }
}

# Data paths
DATA_PATHS = {
    'sample_data': 'data/sample_emails.csv',
    'models_dir': 'models/',
    'logs_dir': 'logs/',
    'plots_dir': 'evaluation_plots/'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/spam_detector.log'
}