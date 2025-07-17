"""Model evaluation and comparison utilities for cyber insurance models."""
from typing import List, Union, Optional, Dict, Any, Set, Tuple
from sklearn.base import BaseEstimator
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from cyber_insurance.utils.constants import OutputPaths
from cyber_insurance.utils.logger import setup_logger

from dataclasses import dataclass, field

# Configure logger
logger = setup_logger(__name__)

@dataclass
class FoldData:
    """Container for fold-specific data and SHAP values."""
    y_test: pd.Series
    y_pred_test: np.ndarray
    fold_model: Optional[BaseEstimator] = None

@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    predictions: np.ndarray
    cv_scores: dict[str, np.ndarray]
    feature_importance: Optional[dict[str, float]] = None
    model: Optional[BaseEstimator] = None
    fold_data: List[FoldData] = field(default_factory=list)

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
        models_with_importance = []
        
        # Filter models with valid feature importance
        for result in self.results:
            if result.feature_importance is not None:
                # Verify that feature importance contains valid numeric data
                try:
                    importance = pd.Series(result.feature_importance)
                    # Check if the data is numeric and not empty
                    if not importance.empty and pd.api.types.is_numeric_dtype(importance):
                        models_with_importance.append(result)
                    else:
                        logger.warning(
                            f"Skipping feature importance plot for {result.model_name}: "
                            f"Non-numeric or empty feature importance data"
                        )
                except Exception as e:
                    logger.warning(
                        f"Skipping feature importance plot for {result.model_name}: {str(e)}"
                    )
        
        if not models_with_importance:
            logger.info("No models with valid feature importance data available for plotting")
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
    
    def plot_metrics_bar(self) -> None:
        """Plot bar charts of key metrics (Accuracy, F1, Cohen's Kappa, MAE) for all models.
        
        This method creates a bar chart comparing all models across four key metrics:
        - Accuracy
        - Weighted F1 Score
        - Cohen's Quadratic Kappa
        - Mean Absolute Error (MAE)
        
        All metrics are calculated using aggregated results from all CV folds.
        """
        # Create output directory if needed
        OutputPaths.create_directories()
        output_dir = OutputPaths.MODEL_EVALUATION_DIR
        
        # Define metrics to plot
        metrics = ['accuracy', 'f1_weighted', 'weighted_mae', 'cem']
        metric_labels = ['Accuracy', 'Weighted F1', 'Weighted MAE', 'CEM']
        
        # Set up the figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Colors for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        # For each metric, create a bar plot
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            # Extract mean scores for each model
            model_names = []
            mean_scores = []
            std_scores = []
            
            for result in self.results:
                model_names.append(result.model_name)
                scores = result.cv_scores.get(f'test_{metric}', np.array([0]))
                mean_scores.append(np.mean(scores))
                std_scores.append(np.std(scores))
            
            # Create bar plot
            bars = ax.bar(
                model_names, 
                mean_scores, 
                yerr=std_scores,
                color=colors,
                alpha=0.7,
                capsize=5
            )
            
            # Add value labels on top of bars
            for bar, score in zip(bars, mean_scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.02,
                    f'{score:.3f}',
                    ha='center', 
                    va='bottom',
                    fontsize=9
                )
            
            # Set title and labels
            ax.set_title(f'{label}', fontsize=12)
            ax.set_ylabel('Score')
            ax.set_ylim(0, max(mean_scores) * 1.2)  # Add some space for labels
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # For MAE, add note that lower is better
            if metric == 'weighted_mae':
                ax.text(
                    0.5, 0.95, 
                    '(Lower is better)', 
                    transform=ax.transAxes,
                    ha='center',
                    fontsize=10,
                    style='italic'
                )
        
        # Add overall title
        fig.suptitle('Model Performance Metrics (Aggregated across CV folds)', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        plt.savefig(
            output_dir / 'model_metrics_comparison.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        logger.info("Generated bar plots for key metrics")
    
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
        if metric == "cem":
            return max(mean_scores.items(), key=lambda x: x[1])[0]
        else:
            return min(mean_scores.items(), key=lambda x: x[1])[0]

    
    def plot_error_distribution(self) -> None:
        """Plot the distribution of absolute errors (|predicted - true|) for each model.
        
        This method creates histograms showing the distribution of absolute errors
        aggregated across all cross-validation folds. For ordinal classification,
        this shows how far off the predictions are from the true classes.
        """
        # Create output directory if needed
        OutputPaths.create_directories()
        output_dir = OutputPaths.MODEL_EVALUATION_DIR
        
        # Get all unique models
        model_names = [result.model_name for result in self.results]
        
        # Set up the figure - one row per model
        fig, axes = plt.subplots(
            len(model_names), 1, 
            figsize=(10, 4 * len(model_names)),
            sharex=True
        )
        
        # Handle case with only one model
        if len(model_names) == 1:
            axes = [axes]
            
        # For each model, create an error distribution plot
        for _, (result, ax) in enumerate(zip(self.results, axes)):
            # Collect true values and predictions across all folds
            y_true = []
            y_pred = []
            
            for fold_data in result.fold_data:
                if hasattr(fold_data, 'y_test') and fold_data.y_test is not None:
                    # Get true values
                    fold_y_true = fold_data.y_test.values
                    
                    # Get predictions directly from fold_data
                    fold_y_pred = fold_data.y_pred_test
                    
                    # Add to overall lists
                    y_true.extend(fold_y_true)
                    y_pred.extend(fold_y_pred)
            
            # Calculate absolute errors
            abs_errors = np.abs(np.array(y_pred) - np.array(y_true))
            
            # Count occurrences of each error value
            error_counts = {}
            for error in abs_errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            # Convert to percentages
            total_samples = len(abs_errors)
            error_percentages = {k: (v / total_samples) * 100 for k, v in error_counts.items()}
            
            # Sort by error value
            sorted_errors = sorted(error_percentages.items())
            error_values = [e[0] for e in sorted_errors]
            error_percentages = [e[1] for e in sorted_errors]
            
            # Create bar chart
            bars = ax.bar(
                error_values,
                error_percentages,
                color='skyblue',
                alpha=0.7,
                width=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Find the maximum percentage for y-axis limit adjustment
            max_percentage = max(error_percentages) if error_percentages else 0
            
            # Add value labels on top of bars
            for bar, percentage in zip(bars, error_percentages):
                height = bar.get_height()
                # Only add labels for bars with significant height
                if height > 0.5:  # Skip very small bars
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height * 1.1,  # Position outside the top of the bar
                        f'{percentage:.1f}%',
                        ha='center', 
                        va='top',
                        fontsize=8,
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
                    )
            
            # Set y-axis limit with enough room for the plot
            ax.set_ylim(0, max_percentage * 1.15)
            
            # Calculate mean absolute error for display
            mae = np.mean(abs_errors)
            
            # Set title and labels
            ax.set_title(f'{result.model_name}: Error Distribution (MAE = {mae:.3f})', fontsize=10)
            ax.set_xlabel('Absolute Error (|Predicted - True|)')
            ax.set_ylabel('Percentage of Samples (%)')
            
            # Set x-ticks to integers
            max_error = int(max(error_values)) if error_values else 0
            ax.set_xticks(range(max_error + 1))
            ax.set_xticklabels(range(max_error + 1))
            
            # Add text with error statistics
            stats_text = (
                f"Mean Abs Error: {mae:.3f}\n"
                f"Median Abs Error: {np.median(abs_errors):.3f}\n"
                f"Perfect Predictions: {error_counts.get(0, 0) / total_samples * 100:.1f}%\n"
                f"Within ±1 Class: {sum(error_counts.get(e, 0) for e in [0, 1]) / total_samples * 100:.1f}%"
            )
            ax.text(
                0.95, 0.95,
                stats_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10
            )
        
        # Add overall title
        fig.suptitle('Error Distribution (|Predicted - True|) Across All CV Folds', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.3)  # Increased spacing between subplots
        
        # Save figure
        plt.savefig(
            output_dir / 'error_distribution.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        logger.info("Generated error distribution plots")
    
    def print_summary(self) -> None:
        """Print summary of model performance."""
        print("\nModel Evaluation Summary:")
        print("-" * 50)
        
        # All metrics to evaluate
        all_metrics = ['mae', 'accuracy', 'f1_weighted', 'weighted_mae', 'cem']
        
        # Print metrics in a single loop
        for metric in all_metrics:
            # Check if at least one model has this metric
            if any(f'test_{metric}' in result.cv_scores for result in self.results):
                print(f"\n{metric.upper()} Scores:")
                for result in self.results:
                    if f'train_{metric}' in result.cv_scores and f'test_{metric}' in result.cv_scores:
                        train_scores = result.cv_scores[f'train_{metric}']
                        test_scores = result.cv_scores[f'test_{metric}']
                        print(
                            f"{result.model_name:20} "
                            f"Train: {np.mean(train_scores):.3f} ± {np.std(train_scores):.3f}  "
                            f"Test: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}"
                        )
                    else:
                        print(f"{result.model_name:20} Not evaluated with {metric}")


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
