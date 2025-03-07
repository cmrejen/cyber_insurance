"""Model evaluation and comparison utilities for cyber insurance models."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cyber_insurance.models.model_trainer import ModelResults
from cyber_insurance.utils.constants import OutputPaths


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
        
        metrics = ['mae', 'accuracy', 'f1_weighted']
        
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
