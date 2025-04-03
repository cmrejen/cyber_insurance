"""Model evaluation and comparison utilities for cyber insurance models."""
from typing import List, Union
from sklearn.base import BaseEstimator
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cyber_insurance.models.model_trainer import ModelResults
from cyber_insurance.utils.constants import OutputPaths

from imblearn.metrics import macro_averaged_mean_absolute_error
class ModelEvaluator:
    """Evaluates and compares model performance."""
    
    def __init__(self, results: List[ModelResults]):
        """Initialize evaluator with model results.
        
        Args:
            results: List of ModelResults to evaluate
        """
        self.results = results
        
    def plot_metric_comparison(self, metric: str) -> None:
        """Plot comparison of model performance for a specific metric.
        
        Args:
            metric: Name of metric to compare (e.g., 'mae', 'accuracy', 'f1_weighted')
        """
        plt.figure(figsize=(10, 6))
        
        data = []
        for result in self.results:
            scores = result.cv_scores[f'test_{metric}']
            data.extend([
                {
                    'Model': result.model_name,
                    'Score': score,
                    'Fold': i
                }
                for i, score in enumerate(scores)
            ])
        
        df = pd.DataFrame(data)
        
        # Create violin plot with individual points
        sns.violinplot(
            data=df,
            x='Model',
            y='Score',
            inner='box',  # Show box plot inside violin
            alpha=0.5
        )
        sns.swarmplot(
            data=df,
            x='Model',
            y='Score',
            color='black',
            alpha=0.5,
            size=4
        )
        
        plt.title(f'Model Comparison: {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Create output directory if needed
        OutputPaths.create_directories()
        output_dir = OutputPaths.MODEL_EVALUATION_DIR
        plt.savefig(
            output_dir / f'model_comparison_{metric}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    def plot_feature_importance(self) -> None:
        """Plot feature importance comparison across models."""
        models_with_importance = [
            r for r in self.results
            if r.feature_importance is not None
        ]
        
        if not models_with_importance:
            return
        
        n_models = len(models_with_importance)
        fig, axes = plt.subplots(
            1, n_models,
            figsize=(6 * n_models, 6)
        )
        if n_models == 1:
            axes = [axes]
        
        for ax, result in zip(axes, models_with_importance):
            importance = pd.Series(result.feature_importance)
            importance.sort_values(ascending=True).plot(
                kind='barh',
                ax=ax
            )
            ax.set_title(f'{result.model_name}\nFeature Importance')
        
        plt.tight_layout()
        
        # Create output directory if needed
        OutputPaths.create_directories()
        output_dir = OutputPaths.MODEL_EVALUATION_DIR
        plt.savefig(
            output_dir / 'feature_importance_comparison.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    def get_best_model(self, metric: str) -> str:
        """Get name of best performing model for a metric.
        
        Args:
            metric: Metric to compare models on
            
        Returns:
            Name of best performing model
        """
        mean_scores = {
            r.model_name: np.mean(r.cv_scores[f'test_{metric}'])
            for r in self.results
        }
        return max(mean_scores.items(), key=lambda x: x[1])[0]
    
    def print_summary(self) -> None:
        """Print summary of model performance."""
        print("\nModel Evaluation Summary:")
        print("-" * 50)
        
        metrics = ['mae', 'accuracy', 'f1_weighted', 'weighted_mae']
        
        for metric in metrics:
            print(f"\n{metric.upper()} Scores:")
            for result in self.results:
                train_scores = result.cv_scores[f'train_{metric}']
                test_scores = result.cv_scores[f'test_{metric}']
                print(
                    f"{result.model_name:20} "
                    f"Train: {np.mean(train_scores):.3f} ± {np.std(train_scores):.3f}  "
                    f"Test: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}"
                )


def weighted_mae_score(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    class_weights: np.ndarray
) -> float:
    """
    Calculate Weighted Mean Absolute Error for ordinal classification.
    
    This implementation combines:
    1. Standard MAE for ordinal regression: |y_true - y_pred|
    2. Custom class weights that are:
       a) Inversely proportional to class frequency
       b) Normalized to sum to n_classes
    
    The formula is:
        WMAE = mean(|y_true - y_pred| * w[y_true])
    where w[i] is normalized to ensure consistent scale across different
    class distributions.
    
    Args:
        y_true: True ordinal labels (1-based indexing)
        y_pred: Predicted ordinal labels (1-based indexing)
        class_weights: Array of weights for each class (from training data)
        
    Returns:
        Weighted MAE score (lower is better).
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()
        
    # Calculate absolute errors (ordinal distance)
    abs_errors = np.abs(y_true - y_pred)
    
    # Weight errors using pre-computed weights from training data
    weighted_errors = abs_errors * class_weights[y_true.astype(int) - 1]
    
    return np.mean(weighted_errors)


def compute_class_weights(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Compute balanced class weights inversely proportional to class frequencies.
    
    The weighting scheme uses:
    1. Basic inverse frequency: n_samples / (n_classes * class_count)
    2. Additional normalization to ensure weights sum to n_classes
    
    This differs from sklearn's compute_class_weight in that we add the 
    normalization step to ensure the weights are on a consistent scale
    regardless of class distribution.
    
    Args:
        y: Target variable with class labels (training data)
        
    Returns:
        Array of weights for each class, normalized to sum to n_classes
    """
    if isinstance(y, pd.Series):
        y = y.to_numpy()
        
    # Count samples per class (skip class 0)
    class_counts = np.bincount(y.astype(int))[1:]
    
    # Compute weights inversely proportional to class frequencies
    n_samples = len(y)
    weights = n_samples / (len(class_counts) * class_counts)
    
    # Normalize weights to sum to n_classes
    weights = weights * (len(class_counts) / weights.sum())
    
    return weights


def cem_score(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    class_counts: np.ndarray
) -> float:
    """
    Calculate Closeness Evaluation Measure (CEM) for ordinal classification.
    
    Args:
    y_true: True ordinal labels (1-based indexing)
    y_pred: Predicted ordinal labels (1-based indexing)
    class_counts: Array of item counts for each class in the training data
    
    Returns:
    CEM score between 0 and 1 (higher is better)
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()
    
    n_classes = len(class_counts)
    N = np.sum(class_counts)  # Total number of training items
    
    # Calculate proximity matrix
    prox_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i <= j:
                prox = -np.log2((class_counts[i] / (2*N)) + np.sum(class_counts[i+1:j+1]) / N)
            else:
                prox = -np.log2((class_counts[j] / (2*N)) + np.sum(class_counts[j+1:i+1]) / N)
            prox_matrix[i, j] = prox
    
    # Calculate numerator and denominator
    numerator = sum(prox_matrix[true-1, pred-1] for true, pred in zip(y_true, y_pred))
    denominator = sum(prox_matrix[true-1, true-1] for true in y_true)
    
    # Return CEM score
    return numerator / denominator
