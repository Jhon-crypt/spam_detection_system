"""
Data loading and preprocessing utilities for spam detection.
Handles CSV data loading, train/test splits, and data validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and preprocessing for spam detection."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data = None
        self.X = None
        self.y = None
        
    def load_csv(self, file_path: str, text_column: str = 'text', 
                 label_column: str = 'label') -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            text_column: Name of the column containing email text
            label_column: Name of the column containing labels
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} samples from {file_path}")
            
            # Validate required columns
            if text_column not in self.data.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
            if label_column not in self.data.columns:
                raise ValueError(f"Label column '{label_column}' not found in data")
                
            # Extract features and labels
            self.X = self.data[text_column].fillna('').astype(str)
            self.y = self.data[label_column]
            
            # Validate data
            self._validate_data()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_from_lists(self, texts: List[str], labels: List[str]) -> None:
        """
        Load data from lists.
        
        Args:
            texts: List of email texts
            labels: List of corresponding labels
            
        Raises:
            ValueError: If lists have different lengths
        """
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
            
        self.data = pd.DataFrame({'text': texts, 'label': labels})
        self.X = pd.Series(texts)
        self.y = pd.Series(labels)
        
        self._validate_data()
        logger.info(f"Loaded {len(texts)} samples from lists")
    
    def _validate_data(self) -> None:
        """Validate the loaded data."""
        if self.X is None or self.y is None:
            raise ValueError("Data not properly loaded")
            
        if len(self.X) == 0:
            raise ValueError("No data samples found")
            
        # Check for valid labels
        unique_labels = self.y.unique()
        logger.info(f"Found labels: {unique_labels}")
        
        # Convert labels to binary if needed
        if set(unique_labels) == {'spam', 'ham'}:
            self.y = self.y.map({'spam': 1, 'ham': 0})
        elif set(unique_labels) == {'spam', 'legitimate'}:
            self.y = self.y.map({'spam': 1, 'legitimate': 0})
        elif not all(label in [0, 1] for label in unique_labels):
            logger.warning(f"Unexpected labels found: {unique_labels}")
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        info = {
            "total_samples": len(self.data),
            "spam_count": sum(self.y == 1),
            "ham_count": sum(self.y == 0),
            "spam_percentage": (sum(self.y == 1) / len(self.y)) * 100,
            "average_text_length": self.X.str.len().mean(),
            "missing_values": self.X.isna().sum()
        }
        
        return info
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42, 
                   stratify: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify the split based on labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If data is not loaded
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before splitting")
            
        stratify_param = self.y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_sample_data(self, output_path: str) -> None:
        """
        Create a sample dataset for testing.
        
        Args:
            output_path: Path to save the sample data
        """
        sample_data = {
            'text': [
                "Congratulations! You've won $1000! Click here to claim your prize now!",
                "Hi John, can we schedule a meeting for tomorrow at 2pm?",
                "URGENT: Your account will be suspended unless you verify immediately!",
                "Thanks for the presentation yesterday. The team found it very helpful.",
                "Free money! No strings attached! Act now!",
                "Please review the attached document and let me know your thoughts.",
                "You are the lucky winner of our lottery! Send your bank details now!",
                "The quarterly report is ready for your review.",
                "CLICK HERE FOR AMAZING DEALS! Limited time offer!",
                "Could you please send me the updated project timeline?",
                "Nigerian prince needs your help to transfer millions!",
                "Meeting room booked for 3pm today for the team standup.",
                "Make money fast working from home! No experience needed!",
                "The client approved the proposal. Great work everyone!",
                "Your computer is infected! Download our antivirus now!",
                "Don't forget about the company picnic this Saturday.",
            ],
            'label': [
                'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 
                'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                'spam', 'ham', 'spam', 'ham'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Sample data created at {output_path}")
        
        return df