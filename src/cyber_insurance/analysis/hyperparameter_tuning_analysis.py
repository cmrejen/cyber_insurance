"""Hyperparameter tuning analysis for ordinal classification models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd
from pathlib import Path
from cyber_insurance.data.ingestion import ICODataIngestion
from cyber_insurance.data.preprocessing import ICODataPreprocessor
from cyber_insurance.data.preprocessing import OrdinalSMOTEResampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from cyber_insurance.models.model_evaluator import (
    cem_score, compute_class_weights, weighted_mae_score
)
from cyber_insurance.models.model_trainer import (
    OrdinalLogistic,
    OrdinalModel,
    RandomForestOrdinal,
    OrdinalNeuralNet,
    CatBoostOrdinal
)
from cyber_insurance.utils.constants import (
    ColumnNames,
    OutputPaths,
    OrdinalMapping,
    ModelParams,
    InputPaths
)
from cyber_insurance.utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm

logger = setup_logger("hyperparameter_tuning_analysis")


@dataclass
class TuningResult:
    """Container for tuning results.
    
    Attributes:
        model_name: Name of the model
        best_params: Best hyperparameters found
        results_df: DataFrame with all tuning results
        confusion_matrices: List of confusion matrices from CV folds for best model
        metric: Scoring metric used for tuning
        best_score: Best score achieved
    """
    model_name: str
    best_params: Dict[str, Any]
    results_df: pd.DataFrame
    confusion_matrices: Optional[List[np.ndarray]] = None
    metric: str = 'cem'
    best_score: float = 0.0


class ModelTuner(ABC):
    """Abstract base class for model tuners."""
    
    def __init__(
        self,
        model_class: Type[OrdinalModel],
        target_col: str,
        cv_folds: int = 5,
        random_state: int = 42,
        metric: str = 'cem'
    ) -> None:
        """Initialize tuner.
        
        Args:
            model_class: Class of model to tune
            target_col: Target column name
            cv_folds: Number of CV folds
            random_state: Random seed
            metric: Scoring metric to optimize, either 'cem' or 'weighted_mae'
        """
        self.model_class = model_class
        self.target_col = target_col
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.metric = metric
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for tuning."""
        pass
    
    def create_model(
        self, 
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> OrdinalModel:
        """Create model instance with parameters.
        
        Args:
            X: Optional feature matrix, required for some models
            y: Optional target variable, required for some models
            **kwargs: Model parameters
            
        Returns:
            Initialized model
        """
        return self.model_class(target_col=self.target_col, **kwargs)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> TuningResult:
        """Tune model hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuning results
        """
        param_grid = self.get_param_grid()
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Initialize results storage
        results = []
        confusion_matrices_list: List[List[np.ndarray]] = []
        
        # Set up stratified k-fold CV
        splits = list(
            StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            ).split(X, y)
        )
        
        # Create progress bar
        total_iterations = len(param_combinations) * self.cv_folds
        pbar = tqdm(total=total_iterations, desc=f"Tuning {self.model_class.__name__}")
        
        try:
            for params in param_combinations:
                fold_scores = []
                fold_confusion_matrices = []
                
                # Cross-validation
                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    try:
                        # Split data
                        X_train = X.iloc[train_idx].copy()
                        y_train = y.iloc[train_idx].copy()
                        X_val = X.iloc[val_idx].copy()
                        y_val = y.iloc[val_idx].copy()
                        
                        # Apply resampling with custom strategy
                        if self.model_class in [OrdinalNeuralNet, RandomForestOrdinal, OrdinalLogistic]:
                            resampler = OrdinalSMOTEResampler(
                                k_neighbors=5,
                                random_state=self.random_state
                            )
                            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
                        else:
                            X_train_resampled, y_train_resampled = X_train, y_train
                        
                        # Train and evaluate model
                        if isinstance(self, OrdinalNeuralNetTuner):
                            model = self.create_model(X=X_train_resampled, y=y_train_resampled, **params)
                        else:
                            model = self.create_model(**params)
                            
                        logger.info(
                            f"Fitting {self.model_class.__name__} with "
                            f"parameters: {params} "
                            f"(Iteration {pbar.n+1}/{total_iterations}, "
                            f"Fold {fold_idx + 1}/{self.cv_folds})"
                        )
                        model.fit(X_train_resampled, y_train_resampled)
                        
                        # Evaluate on original validation data
                        y_pred = model.predict(X_val)
                        
                        # Calculate scoring metric
                        if self.metric == 'cem':
                            # Calculate class probabilities from training fold
                            train_counts = np.bincount(y_train)[1:]
                            score = cem_score(
                                y_val,
                                y_pred,
                                class_counts=train_counts
                            )
                        elif self.metric == 'weighted_mae':
                            # Compute weights from training fold
                            class_weights = compute_class_weights(y_train)
                            score = weighted_mae_score(y_val, y_pred, class_weights)
                        else:
                            raise ValueError(f"Unknown metric: {self.metric}. Use 'cem' or 'weighted_mae'")
                        
                        logger.info(
                            f"======================= {self.metric.upper()} score: {score:.3f} ======================="
                        )
                        fold_scores.append(score)
                        
                        # Calculate confusion matrix
                        cm = confusion_matrix(y_val, y_pred)
                        fold_confusion_matrices.append(cm)
                        
                        # Update progress bar
                        pbar.update(1)
                    
                    except Exception as e:
                        logger.error(f"Error in CV fold: {e}")
                        fold_scores.append(float('-inf'))  # Minimize for bad params
                        fold_confusion_matrices.append(np.zeros((len(np.unique(y)), len(np.unique(y)))))
                
                # Record results
                result = {
                    **params,
                    f'{self.metric}_score': np.mean(fold_scores),
                    f'{self.metric}_std': np.std(fold_scores)
                }
                results.append(result)
                confusion_matrices_list.append(fold_confusion_matrices)
        
        finally:
            pbar.close()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best parameters (maximize CEM or minimize weighted MAE)
        if self.metric == 'cem':
            best_idx = results_df[f'{self.metric}_score'].argmax()
        else:
            best_idx = results_df[f'{self.metric}_score'].argmin()
        best_params = {
            param: results_df.loc[best_idx, param]
            for param in param_grid.keys()
        }
        
        # Log results
        logger.info(
            f"Best {self.model_class.__name__} parameters:\n"
            + "\n".join(
                f"{k}: {v}" for k, v in best_params.items()
            )
        )
        logger.info(
            f"Mean {self.metric.upper()} score: {results_df.loc[best_idx, f'{self.metric}_score']:.3f} "
            f"± {results_df.loc[best_idx, f'{self.metric}_std']:.3f}"
        )
        
        return TuningResult(
            model_name=self.model_class.__name__,
            best_params=best_params,
            results_df=results_df,
            confusion_matrices=confusion_matrices_list[best_idx],
            metric=self.metric,
            best_score=results_df.loc[best_idx, f'{self.metric}_score']
        )
    
    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]


class RandomForestTuner(ModelTuner):
    """Hyperparameter tuner for Random Forest model."""
    
    def __init__(
        self,
        target_col: str,
        cv_folds: int = 5,
        random_state: int = 42,
        metric: str = 'cem'
    ) -> None:
        """Initialize RF tuner."""
        super().__init__(
            model_class=RandomForestOrdinal,
            target_col=target_col,
            cv_folds=cv_folds,
            random_state=random_state,
            metric=metric
        )
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get RF parameter grid."""
        return {
            'n_estimators': ModelParams.RF_N_ESTIMATORS,
            'min_samples_leaf': ModelParams.RF_MIN_SAMPLES_LEAF,
            'max_features': ModelParams.RF_MAX_FEATURES
        }
    
    def create_model(self, **kwargs: Any) -> RandomForestOrdinal:
        """Create RF model instance."""
        return self.model_class(
            target_col=self.target_col,
            random_state=self.random_state,
            **kwargs
        )


class OrdinalLogisticTuner(ModelTuner):
    """Hyperparameter tuner for Ordinal Logistic model."""
    
    def __init__(
        self,
        target_col: str,
        cv_folds: int = 5,
        metric: str = 'cem'
    ) -> None:
        """Initialize OL tuner."""
        super().__init__(
            model_class=OrdinalLogistic,
            target_col=target_col,
            cv_folds=cv_folds,
            metric=metric
        )
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get OL parameter grid."""
        return {
            'alpha': ModelParams.OL_ALPHA
        }
    
    def create_model(self, **kwargs: Any) -> OrdinalLogistic:
        """Create OL model instance."""
        return self.model_class(
            target_col=self.target_col,
            **kwargs
        )

class OrdinalNeuralNetTuner(ModelTuner):
    """Hyperparameter tuner for PyTorch Ordinal model."""
    
    def __init__(
        self,
        target_col: str,
        cv_folds: int = 5,
        random_state: int = 42,
        metric: str = 'cem'
    ) -> None:
        """Initialize PO tuner."""
        super().__init__(
            model_class=OrdinalNeuralNet,
            target_col=target_col,
            cv_folds=cv_folds,
            random_state=random_state,
            metric=metric
        )
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get PO parameter grid."""
        return {
            "hidden_layer_sizes": ModelParams.PTORDINAL_HIDDEN_LAYER_SIZES,
            "lr": ModelParams.PTORDINAL_LR,
            "epochs": ModelParams.PTORDINAL_EPOCHS,
            "batch_size": ModelParams.PTORDINAL_BATCH_SIZE
        }
    
    def create_model(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any
    ) -> OrdinalNeuralNet:
        """Create PO model instance.
        
        Args:
            X: Feature matrix (preprocessed with dummy/ordinal encoding)
            y: Target variable (values 1-6 representing data subject ranges)
            **kwargs: Model parameters
            
        Returns:
            Initialized neural network model
            
        Notes:
            Target values are automatically shifted to start from 0 in the
            OrdinalDataset class to match CORAL's requirements.
        """
        if X is None or y is None:
            raise ValueError("X and y are required for neural network initialization")
            
        # All features are numerical after preprocessing
        num_numerical_features = X.shape[1]
        
        # Get total number of classes (K)
        unique_labels = np.sort(y.unique())
        num_classes = len(unique_labels)  # K classes total
        
        return self.model_class(
            target_col=self.target_col,
            num_numerical_features=num_numerical_features,
            num_classes=num_classes,  # Pass total number of classes (K)
            **kwargs
        )


class CatBoostOrdinalTuner(ModelTuner):
    """Hyperparameter tuner for CatBoost Ordinal model."""
    
    def __init__(
        self,
        target_col: str,
        cv_folds: int = 5,
        random_state: int = 42,
        metric: str = 'cem'
    ) -> None:
        """Initialize CBO tuner."""
        super().__init__(
            model_class=CatBoostOrdinal,
            target_col=target_col,
            cv_folds=cv_folds,
            random_state=random_state,
            metric=metric
        )
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get CBO parameter grid."""
        return {
            "iterations": ModelParams.CB_ITERATIONS,
            "learning_rate": ModelParams.CB_LEARNING_RATE,
            "depth": ModelParams.CB_DEPTH,
            "l2_leaf_reg": ModelParams.CB_L2_LEAF_REG
        }
    
    def create_model(self, **kwargs: Any) -> CatBoostOrdinal:
        """Create CBO model instance."""
        return self.model_class(
            target_col=self.target_col,
            **kwargs
        )


class ModelTuningVisualizer:
    """Visualizer for model tuning results and performance metrics."""

    _MODEL_DIRS = {
        'CatBoostOrdinal': OutputPaths.CATBOOST_TUNING_DIR,
        'OrdinalNeuralNet': OutputPaths.NEURAL_NET_TUNING_DIR,
        'RandomForestOrdinal': OutputPaths.RANDOM_FOREST_TUNING_DIR,
        'OrdinalLogistic': OutputPaths.ORDINAL_LOGISTIC_TUNING_DIR
    }

    def __init__(self, output_dir: Path = OutputPaths.HYPERPARAMETER_TUNING_DIR) -> None:
        """Initialize visualizer.
        
        Args:
            output_dir: Base directory for saving plots
        """
        # Ensure output directory exists
        OutputPaths.create_directories()
        self.output_dir = output_dir
        
    def _get_model_dir(self, model_name: str) -> Path:
        """Get model-specific output directory.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model-specific directory
        """
        return self._MODEL_DIRS.get(model_name, self.output_dir)

    def _save_plot(self, fig: plt.Figure, model_name: str, plot_type: str) -> None:
        """Save plot to model-specific directory.
        
        Args:
            fig: Figure to save
            model_name: Name of the model
            plot_type: Type of plot (e.g., 'tuning', 'confusion_matrix')
        """
        save_path = self._get_model_dir(model_name) / f"{plot_type}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_tuning_results(
        self,
        results: List[TuningResult],
    ) -> None:
        """Plot tuning results for all models.
        
        Args:
            results: List of tuning results
        """
        for result in results:
            self._plot_model_tuning_results(result)
            if result.confusion_matrices:
                self.plot_confusion_matrix(result)

    def _plot_model_tuning_results(self, result: TuningResult) -> None:
        """Plot hyperparameter tuning results for a specific model.
        
        Args:
            result: Tuning results for the model
        """
        if result.model_name == 'RandomForestOrdinal':
            self._plot_rf_tuning_results(result)
        elif result.model_name == 'OrdinalLogistic':
            self._plot_ol_tuning_results(result)
        elif result.model_name == 'OrdinalNeuralNet':
            self._plot_pytorch_ordinal_tuning_results(result)
        elif result.model_name == 'CatBoostOrdinal':
            self._plot_catboost_tuning_results(result)
        else:
            logger.warning(f"No visualization available for {result.model_name}")

    def _plot_rf_tuning_results(self, result: TuningResult) -> None:
        """Plot Random Forest tuning results using heatmaps.
        
        Args:
            result: Random Forest tuning results
        """
        n_estimators = result.results_df['n_estimators'].unique()
        fig, axes = plt.subplots(
            1, len(n_estimators),
            figsize=(7*len(n_estimators), 5)
        )
        
        # Handle single subplot case
        if len(n_estimators) == 1:
            axes = [axes]
        
        for idx, n_est in enumerate(n_estimators):
            subset = result.results_df[
                result.results_df['n_estimators'] == n_est
            ]
            pivot = subset.pivot(
                index='min_samples_leaf',
                columns='max_features',
                values=f'{result.metric}_score'
            )
            
            # Plot heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                ax=axes[idx],
                cmap='YlOrRd'
            )
            axes[idx].set_title(f'n_estimators={n_est}')
            axes[idx].set_xlabel('max_features')
            axes[idx].set_ylabel('min_samples_leaf')
        
        if result.metric == 'cem':
            best_metric_text = f'Best {result.metric.upper()} score: {result.results_df[f"{result.metric}_score"].max():.3f}'
        else:
            best_metric_text = f'Best {result.metric.upper()} score: {result.results_df[f"{result.metric}_score"].min():.3f}'
        
        plt.suptitle(
            f'{result.model_name} Hyperparameter Tuning Results\n'
            f'{best_metric_text}'
        )
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, result.model_name, 'tuning')

    def _plot_ol_tuning_results(self, result: TuningResult) -> None:
        """Plot Ordinal Logistic tuning results using line plot.
        
        Args:
            result: Ordinal Logistic tuning results
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Plot regularization vs CEM score
        sns.lineplot(
            data=result.results_df,
            x='alpha',
            y=f'{result.metric}_score',
            ax=ax,
            marker='o'
        )
        
        # Add error bars
        ax.fill_between(
            result.results_df['alpha'],
            result.results_df[f'{result.metric}_score'] - result.results_df[f'{result.metric}_std'],
            result.results_df[f'{result.metric}_score'] + result.results_df[f'{result.metric}_std'],
            alpha=0.2
        )
        
        ax.set_title(
            f'{result.model_name} Hyperparameter Tuning Results\n'
            f'Best {result.metric.upper()} score: {result.results_df[f"{result.metric}_score"].max():.3f}',
            fontsize=14
        )
        ax.set_xlabel('Regularization (alpha)')
        ax.set_ylabel(f'{result.metric.upper()} Score')
        ax.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, result.model_name, 'tuning')

    def _plot_pytorch_ordinal_tuning_results(self, result: TuningResult) -> None:
        """Plot PyTorch Ordinal tuning results using line plot.
        
        Args:
            result: PyTorch Ordinal tuning results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by hidden_layer_sizes and plot learning rate vs CEM score
        for hidden_layer_sizes, group in result.results_df.groupby('hidden_layer_sizes'):
            sns.lineplot(
                data=group,
                x='lr',
                y=f'{result.metric}_score',
                ax=ax,
                marker='o',
                label=f'Hidden Layers: {hidden_layer_sizes}'
            )
        
        # Customize plot
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(f'{result.metric.upper()} Score')
        ax.set_title(
            f'{result.model_name} Hyperparameter Tuning Results\n'
            f'Best {result.metric.upper()} score: {result.results_df[f"{result.metric}_score"].max():.3f}',
            fontsize=14
        )
        ax.grid(True, which='both', linestyle='--')
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, result.model_name, 'tuning')

    def _plot_catboost_tuning_results(self, result: TuningResult) -> None:
        """Plot CatBoost tuning results using heatmap.
        
        Args:
            result: CatBoost tuning results
        """
        fig, axes = plt.subplots(
            1, len(ModelParams.CB_DEPTH),
            figsize=(7 * len(ModelParams.CB_DEPTH), 5),
            sharey=True
        )
        
        for idx, depth in enumerate(ModelParams.CB_DEPTH):
            # Pivot data for heatmap
            pivot = result.results_df[
                result.results_df["depth"] == depth
            ].pivot_table(
                index='l2_leaf_reg',
                columns='learning_rate',
                values=f'{result.metric}_score'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                ax=axes[idx],
                cmap='YlOrRd'
            )
            axes[idx].set_title(f'Depth={depth}')
            axes[idx].set_xlabel('Learning Rate')
            axes[idx].set_ylabel('L2 Regularization')
        
        if result.metric == 'cem':
            best_metric_text = f'Best {result.metric.upper()} score: {result.results_df[f"{result.metric}_score"].max():.3f}'
        else:
            best_metric_text = f'Best {result.metric.upper()} score: {result.results_df[f"{result.metric}_score"].min():.3f}'
        
        plt.suptitle(
            f'{result.model_name} Hyperparameter Tuning Results\n'
            f'{best_metric_text}',
            fontsize=14
        )
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, result.model_name, 'tuning')

    def plot_confusion_matrix(self, result: TuningResult) -> None:
        """Plot confusion matrix with ordinal-specific metrics.
        
        Args:
            result: Tuning result containing confusion matrices
        """
        if not result.confusion_matrices:
            logger.warning(f"No confusion matrices available for {result.model_name}")
            return
            
        # Get filtered class labels (i.e. post-imputation)
        class_labels = OrdinalMapping.filtered_names(OrdinalMapping.NO_SUBJECTS_AFFECTED)
        
        # Average confusion matrices across folds
        mean_cm = np.mean(result.confusion_matrices, axis=0)
        
        # Calculate metrics
        row_sums = mean_cm.sum(axis=1, keepdims=True)
        cm_normalized = mean_cm / row_sums
        
        # Calculate ordinal-specific metrics
        n_classes = mean_cm.shape[0]
        ordinal_errors = np.zeros(n_classes)
        for i in range(n_classes):
            # Weight errors by distance from true class
            for j in range(n_classes):
                if i != j:
                    ordinal_errors[i] += cm_normalized[i,j] * abs(i - j)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 2, width_ratios=[2, 1])
        
        # Plot normalized confusion matrix
        ax1 = plt.subplot(gs[:, 0])
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax1
        )
        ax1.set_title(
            f'Normalized Confusion Matrix\n'
            f'Mean across {len(result.confusion_matrices)} folds'
        )
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # Plot ordinal error distribution
        ax2 = plt.subplot(gs[0, 1])
        sns.barplot(
            x=np.arange(len(class_labels)),
            y=ordinal_errors,
            ax=ax2
        )
        ax2.set_title('Average Ordinal Error by Class')
        ax2.set_xticks([])
        ax2.set_ylabel('Mean Ordinal Error')
        
        # Add class performance metrics
        ax3 = plt.subplot(gs[1, 1])
        ax3.axis('off')
        
        # Calculate per-class metrics
        metrics_text = "Class-wise Performance:\n\n"
        for i, label in enumerate(class_labels):
            precision = cm_normalized[i, i]
            total_error = ordinal_errors[i]
            metrics_text += (
                f"{label}:\n"
                f"  Accuracy: {precision:.2f}\n"
                f"  Ordinal Error: {total_error:.2f}\n\n"
            )
        
        ax3.text(
            0, 1, metrics_text,
            va='top', ha='left',
            fontsize=10,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.8
            )
        )
        
        best_metric_text = f'Best {result.metric.upper()} score: {result.best_score:.3f}'
        
        plt.suptitle(
            f'{result.model_name} Performance Analysis\n'
            f'{best_metric_text}',
            fontsize=14,
            y=1.02
        )
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, result.model_name, 'confusion_matrix')


if __name__ == '__main__':
    # Set data path and target
    data_file = Path(InputPaths.ICO_BREACH_DATA)
    target_col = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value

    # Step 1: Data Ingestion
    logger.info("Starting data ingestion...")
    ingestion = ICODataIngestion()
    df = ingestion.load_data(data_file)
    
    # Step 2: Preprocessing
    logger.info("Preprocessing data...")
    preprocessor = ICODataPreprocessor()
    processed_df = preprocessor.preprocess(df)
    
    # Step 3: Prepare features and target
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    
    # Step 4: Initialize visualizer and tuners
    visualizer = ModelTuningVisualizer()
    tuners = [
        OrdinalLogisticTuner(target_col=target_col, metric='weighted_mae'),
        RandomForestTuner(target_col=target_col, metric='weighted_mae'),
        CatBoostOrdinalTuner(target_col=target_col, metric='weighted_mae'),
        OrdinalNeuralNetTuner(target_col=target_col, metric='weighted_mae'),
    ]
    
    # Step 5: Tune models and collect results
    results = []
    
    for tuner in tuners:
        logger.info(f"Tuning {tuner.model_class.__name__}...")
        result = tuner.tune(X, y)
        results.append(result)
        logger.info(f"Tuning {tuner.model_class.__name__} finished.")
    
    # Step 6: Plot combined results
    visualizer.plot_tuning_results(results)
