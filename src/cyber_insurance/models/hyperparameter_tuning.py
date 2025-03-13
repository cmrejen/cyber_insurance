"""Hyperparameter tuning utilities for model optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import orf
from sklearn.model_selection import StratifiedKFold

from cyber_insurance.utils.constants import ModelParams, OutputPaths
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("hyperparameter_tuning")


class TuningError(Exception):
    """Base exception for tuning errors."""


@dataclass(frozen=True)
class TuningResults:
    """Results from hyperparameter tuning."""
    best_params: Dict[str, Any]
    cv_results: pd.DataFrame
    best_score: float


@dataclass(frozen=True)
class RandomForestResults(TuningResults):
    """Results from Ordered Forest tuning."""
    performance_scores: Dict[str, List[float]]
    param_grid: Dict[str, List[Any]]


class BaseTuner(ABC):
    """Abstract base class for hyperparameter tuning."""
    
    def __init__(
        self,
        cv_folds: int = 5,
        scoring: str = 'neg_mean_squared_error',  # Better for ordinal
        random_state: int = 42,
        n_jobs: int = -1
    ) -> None:
        """Initialize tuner.
        
        Args:
            cv_folds: CV folds
            scoring: Metric
            random_state: Seed
            n_jobs: Parallel jobs
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid."""
        pass
    
    @abstractmethod
    def create_model(self, **kwargs: Any) -> Any:
        """Create model instance."""
        pass


class RandomForestTuner(BaseTuner):
    """Ordered Forest tuner with performance tracking."""
    
    def __init__(
        self,
        n_estimators_range: Optional[List[int]] = None,
        min_samples_leaf_range: Optional[List[int]] = None,
        max_features_range: Optional[List[int]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize ORF tuner.
        
        Args:
            n_estimators_range: Number of trees to try
            min_samples_leaf_range: Min samples per leaf to try
            max_features_range: Number of features to try
            **kwargs: Additional args for BaseTuner
        """
        super().__init__(**kwargs)
        self.n_estimators_range = (
            n_estimators_range or ModelParams.RF_N_ESTIMATORS
        )
        self.min_samples_leaf_range = (
            min_samples_leaf_range or ModelParams.RF_MIN_SAMPLES_LEAF
        )
        self.max_features_range = (
            max_features_range or ModelParams.RF_MAX_FEATURES
        )
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get ORF parameter grid."""
        return {
            'n_estimators': self.n_estimators_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range
        }
    
    def create_model(self, **kwargs: Any) -> orf.OrderedForest:
        """Create ORF with fixed params."""
        return orf.OrderedForest(
            replace=True,
            honesty=False,
            random_state=self.random_state,
            **kwargs
        )
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> RandomForestResults:
        """Tune ORF and track performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Optional custom grid
            
        Returns:
            Tuning results with best parameters
            
        Raises:
            TuningError: If tuning fails
        """
        try:
            if param_grid is None:
                param_grid = self.get_param_grid()
            
            # Calculate performance for each parameter combination
            scores = self._calculate_performance(X, y, param_grid)
            
            # Find best parameters
            df_scores = pd.DataFrame(scores)
            best_idx = df_scores['mae'].argmin()
            best_params = {
                'n_estimators': int(df_scores.loc[best_idx, 'n_estimators']),
                'min_samples_leaf': int(df_scores.loc[best_idx, 'min_samples_leaf']),
                'max_features': int(df_scores.loc[best_idx, 'max_features'])
            }
            best_score = df_scores.loc[best_idx, 'mae']
            
            logger.info(
                f"Best parameters: {best_params}\n"
                f"Best score: {best_score:.4f}"
            )
            
            self._plot_performance_surface(scores)
            
            return RandomForestResults(
                best_params=best_params,
                cv_results=df_scores,
                best_score=best_score,
                performance_scores=scores,
                param_grid=param_grid
            )
            
        except Exception as e:
            logger.error(f"ORF tuning failed: {e}")
            raise TuningError(f"Failed to tune ORF: {e}") from e
    
    def _calculate_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, List[float]]:
        """Calculate performance metrics using CV.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid to search
            
        Returns:
            Dictionary with performance metrics. Uses ORF's built-in performance
            method which calculates MSE based on ordered probability predictions.
        """
        scores: Dict[str, List[float]] = {
            'n_estimators': [],
            'min_samples_leaf': [],
            'max_features': [],
            'mae': []  # Mean absolute error from ORF performance
        }
        
        # Convert to numpy arrays with proper types
        X_values = np.asarray(X, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.int32).reshape(-1)  # Ensure 1D array
        
        # Create CV splits
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Store CV splits to avoid index errors
        splits = list(cv.split(X_values, y_values))
        
        for n_est in param_grid['n_estimators']:
            for min_samples in param_grid['min_samples_leaf']:
                for max_feat in param_grid['max_features']:
                    cv_scores = []
                    
                    # Cross-validation
                    for train_idx, val_idx in splits:
                        # Create copies to avoid index errors
                        X_train = X_values[train_idx].copy()
                        y_train = y_values[train_idx].copy()
                        X_test = X_values[val_idx].copy()
                        y_test = y_values[val_idx].copy()
                        
                        # Train model
                        try:
                            rf = self.create_model(
                                n_estimators=int(n_est),
                                min_samples_leaf=int(min_samples),
                                max_features=int(max_feat)
                            )
                            
                            # Fit model
                            rf.fit(X_train, y_train)
                            y_pred_test = rf.predict(X_test, prob=False)["predictions"]
                            # Get MAE using ORF's performance method
                            mae = mean_absolute_error(y_test, y_pred_test)
                                
                        except Exception as e:
                            logger.error(f"Error in CV fold: {e}")
                            mae = float('inf')  # Penalize failed configurations
                            
                        cv_scores.append(mae)
                    
                    # Record average performance
                    scores['n_estimators'].append(n_est)
                    scores['min_samples_leaf'].append(min_samples)
                    scores['max_features'].append(max_feat)
                    scores['mae'].append(np.mean(cv_scores))
        
        return scores
    
    def _plot_performance_surface(
        self,
        scores: Dict[str, List[float]]
    ) -> None:
        """Plot performance surface.
        
        Args:
            scores: Performance metrics
        """
        OutputPaths.create_directories()
        output_dir = OutputPaths.MODEL_EVALUATION_DIR
        
        df = pd.DataFrame(scores)
        
        # Create subplots for each min_samples_leaf value
        unique_min_samples = sorted(df['min_samples_leaf'].unique())
        n_plots = len(unique_min_samples)
        fig, axes = plt.subplots(
            1, n_plots,
            figsize=(6 * n_plots, 5),
            squeeze=False
        )
        
        for i, min_samples in enumerate(unique_min_samples):
            df_subset = df[df['min_samples_leaf'] == min_samples]
            pivot = df_subset.pivot(
                index='max_features',
                columns='n_estimators',
                values='mae'
            )
            
            ax = axes[0, i]
            im = ax.imshow(pivot, cmap='viridis_r', aspect='auto')
            plt.colorbar(im, ax=ax, label='Mean Absolute Error')
            
            ax.set_xlabel('Number of Estimators')
            ax.set_ylabel('Max Features')
            ax.set_title(f'Min Samples Leaf = {min_samples}')
            
            # Set tick labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
        
        plt.suptitle('Ordered Forest Performance\nParameter Sensitivity')
        plt.tight_layout()
        
        plt.savefig(
            output_dir / 'orf_performance_surface.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
