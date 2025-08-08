"""
Model evaluation module for spam detection.
Provides comprehensive evaluation metrics and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for spam detection."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.metrics = {}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC/PR curves)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # True/False Positives/Negatives
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC AUC if probabilities are provided
        if y_proba is not None:
            if len(y_proba.shape) > 1:
                # If probabilities for both classes, use positive class
                y_proba = y_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)
        
        self.metrics = metrics
        
        logger.info(f"Model Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   target_names: List[str] = None) -> None:
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for the classes
        """
        if target_names is None:
            target_names = ['Ham', 'Spam']
            
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\nDetailed Classification Report:")
        print("=" * 50)
        print(report)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str] = None, 
                             normalize: bool = False,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for the classes
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = ['Ham', 'Spam']
            
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
            
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
            
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR curve (AP = {ap_score:.2f})')
        
        # Add baseline (random classifier)
        baseline = sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='red', linestyle='--', 
                  label=f'Random classifier (AP = {baseline:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
            
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               feature_importance: np.ndarray,
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            feature_importance: Importance scores
            top_n: Number of top features to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get top features
        indices = np.argsort(np.abs(feature_importance))[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in top_importance]
        
        bars = ax.barh(range(len(top_features)), top_importance, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}',
                   ha='left' if value >= 0 else 'right', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            
        return fig
    
    def create_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               feature_names: Optional[List[str]] = None,
                               feature_importance: Optional[np.ndarray] = None,
                               save_dir: Optional[str] = None) -> Dict:
        """
        Create comprehensive evaluation report with all metrics and plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            feature_names: Names of features
            feature_importance: Feature importance scores
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with all evaluation results
        """
        # Evaluate model
        metrics = self.evaluate_model(y_true, y_pred, y_proba)
        
        # Print classification report
        self.print_classification_report(y_true, y_pred)
        
        # Create plots
        plots = {}
        
        # Confusion matrix
        cm_path = f"{save_dir}/confusion_matrix.png" if save_dir else None
        plots['confusion_matrix'] = self.plot_confusion_matrix(
            y_true, y_pred, save_path=cm_path)
        
        if y_proba is not None:
            # ROC curve
            roc_path = f"{save_dir}/roc_curve.png" if save_dir else None
            plots['roc_curve'] = self.plot_roc_curve(
                y_true, y_proba, save_path=roc_path)
            
            # PR curve
            pr_path = f"{save_dir}/pr_curve.png" if save_dir else None
            plots['pr_curve'] = self.plot_precision_recall_curve(
                y_true, y_proba, save_path=pr_path)
        
        if feature_names is not None and feature_importance is not None:
            # Feature importance
            feat_path = f"{save_dir}/feature_importance.png" if save_dir else None
            plots['feature_importance'] = self.plot_feature_importance(
                feature_names, feature_importance, save_path=feat_path)
        
        return {
            'metrics': metrics,
            'plots': plots
        }