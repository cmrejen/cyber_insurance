"""Model training and evaluation framework for cyber insurance data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mord import LogisticIT
import orf
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (mean_absolute_error, accuracy_score, f1_score)

from cyber_insurance.models.hyperparameter_tuning import RandomForestTuner
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("model_trainer")


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    predictions: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None


class OrdinalModel(ABC):
    """Base class for ordinal models."""
    
    def __init__(self, target_col: str) -> None:
        """Initialize model.
        
        Args:
            target_col: Name of target column
        """
        self.target_col = target_col
        self.model: Optional[object] = None
        self.feature_names: List[str] = []
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model to data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        pass


class RandomForestOrdinal(OrdinalModel):
    """Ordinal Random Forest using ORF package."""
    
    def __init__(
        self,
        target_col: str,
        n_estimators: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        max_features: Optional[int] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        inference: bool = True,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        sample_fraction: float = 0.5
    ) -> None:
        """Initialize model.
        
        Args:
            target_col: Target column name
            n_estimators: Number of trees (tuned if None)
            min_samples_leaf: Minimum samples per leaf (tuned if None)
            max_features: Number of features to consider (tuned if None)
            cv_folds: Cross-validation folds
            random_state: Random seed
            inference: Whether to enable inference mode
            honesty: Whether to use honest splitting
            honesty_fraction: Fraction of data for honest estimation
            sample_fraction: Fraction of data for each tree
        """
        super().__init__(target_col)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.inference = inference
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.sample_fraction = sample_fraction
        self.model: Optional[orf.OrderedForest] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model with hyperparameter tuning if needed."""
        try:
            self.feature_names = list(X.columns)
            
            # Ensure data is properly formatted
            X_clean = X.copy()
            y_clean = y.astype(int)  # Ensure integer labels
            
            # Drop any rows with missing values
            valid_mask = ~(X_clean.isna().any(axis=1) | y_clean.isna())
            X_clean = X_clean[valid_mask]
            y_clean = y_clean[valid_mask]
            
            # Convert to proper types for ORF
            X_clean = X_clean.astype(np.float64)
            
            # Tune hyperparameters if needed
            if any(p is None for p in [
                self.n_estimators,
                self.min_samples_leaf,
                self.max_features
            ]):
                tuner = RandomForestTuner(
                    cv_folds=self.cv_folds,
                    random_state=self.random_state
                )
                results = tuner.tune(X_clean, y_clean)
                self.n_estimators = int(results.best_params['n_estimators'])
                self.min_samples_leaf = int(results.best_params['min_samples_leaf'])
                self.max_features = int(results.best_params['max_features'])
                logger.info(
                    f"Tuned parameters: n_estimators={self.n_estimators}, "
                    f"min_samples_leaf={self.min_samples_leaf}, "
                    f"max_features={self.max_features}"
                )
            
            # Initialize and train model
            self.model = orf.OrderedForest(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                replace=False,
                sample_fraction=self.sample_fraction,
                honesty=self.honesty,
                honesty_fraction=self.honesty_fraction,
                inference=self.inference,
                random_state=self.random_state
            )
            
            # Convert data to numpy arrays with proper types
            X_values = np.asarray(X_clean.values, dtype=np.float64)
            y_values = np.asarray(y_clean.values, dtype=np.int32)
            
            self.model.fit(X_values, y_values)
            
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            raise ValueError(f"Model fitting failed: {e}") from e
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")
        
        try:
            # Handle missing values for prediction
            X_clean = X.copy()
            X_clean = X_clean.fillna(X_clean.mean())  # Simple imputation
            
            # Convert to proper types for ORF
            X_clean = X_clean.astype(np.float64)
            X_values = np.asarray(X_clean.values, dtype=np.float64)
            
            # Get predictions and handle different return types
            y_pred = self.model.predict(X_values)
            
            # Convert predictions to array
            if isinstance(y_pred, dict):
                y_pred = np.array([y_pred[i] for i in range(len(X))])
            elif isinstance(y_pred, list):
                y_pred = np.array(y_pred)
                
            return y_pred.reshape(-1)  # Ensure 1D array
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to generate predictions: {e}") from e
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")
        
        try:
            # Handle missing values for prediction
            X_clean = X.copy()
            X_clean = X_clean.fillna(X_clean.mean())  # Simple imputation
            
            # Convert to proper types for ORF
            X_clean = X_clean.astype(np.float64)
            X_values = np.asarray(X_clean.values, dtype=np.float64)
            
            # Get probabilities and ensure proper shape
            probs = self.model.predict_proba(X_values)
            if isinstance(probs, dict):
                probs = np.array([probs[i] for i in range(len(X))])
                
            return probs
            
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise ValueError(f"Failed to generate probabilities: {e}") from e
    
    def predict_proba_with_intervals(
        self,
        X: pd.DataFrame,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predicted probabilities with confidence intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level (e.g., 0.05 for 95% CI)
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")
            
        if not self.inference:
            raise ValueError("Model must be trained with inference=True")
        
        try:
            # Handle missing values for prediction
            X_clean = X.copy()
            X_clean = X_clean.fillna(X_clean.mean())  # Simple imputation
            
            # Convert to proper types for ORF
            X_clean = X_clean.astype(np.float64)
            X_values = np.asarray(X_clean.values, dtype=np.float64)
            
            # Get predictions
            y_pred = self.predict(X)
            
            # Get variance estimates
            variance = self.model.prediction_variance(X_values)
            if isinstance(variance, dict):
                variance = np.array([variance[i] for i in range(len(X))])
            
            # Calculate confidence intervals
            margin = 1.96 * np.sqrt(variance)  # 95% CI
            lower_bound = np.clip(y_pred - margin, 0, None)
            upper_bound = np.clip(y_pred + margin, None, len(np.unique(y_pred)) - 1)
            
            return y_pred, lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Prediction with intervals failed: {e}")
            raise ValueError(
                f"Failed to generate predictions with intervals: {e}"
            ) from e
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be fitted before getting importance")
            
        try:
            importances = self.model.feature_importance()
            if importances is None:
                return None
                
            if isinstance(importances, dict):
                importances = np.array([importances[i] for i in range(len(self.feature_names))])
                
            return dict(zip(self.feature_names, importances))
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return None


class XGBoostOrdinal(OrdinalModel):
    """XGBoost for ordinal classification."""
    
    def __init__(
        self,
        target_col: str,
        n_estimators: int = 100,
        max_depth: Optional[int] = None
    ):
        super().__init__(target_col)
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            objective='multi:softmax',
            eval_metric='mlogloss'
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))


class OrdinalLogistic(OrdinalModel):
    """Ordinal regression using mord's LogisticIT model."""
    
    def __init__(self, target_col: str) -> None:
        super().__init__(target_col)
        self.model = LogisticIT()
        self.feature_names: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            self.feature_names = X.columns.tolist()
            self.model.fit(X, y)
        except Exception as e:
            logger.error(f"Failed to fit ordinal model: {e}")
            raise ValueError(f"Model fitting failed: {e}") from e
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model must be fitted before predicting")
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to predict: {e}") from e
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not hasattr(self.model, 'coef_') or not self.feature_names:
            return None
        try:
            importance = np.abs(self.model.coef_)
            return dict(zip(self.feature_names, importance))
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return None


class OrdinalNeuralNet(OrdinalModel):
    """Neural Network for ordinal classification."""
    
    def __init__(
        self,
        target_col: str,
        input_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__(target_col)
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Ordinal output
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.feature_names: Optional[List[str]] = None
    
    def _to_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        """Convert DataFrame to tensor."""
        self.feature_names = list(X.columns)
        return torch.FloatTensor(X.values).to(self.device)
    
    def _integrated_gradients(
        self,
        X: torch.Tensor,
        n_steps: int = 50
    ) -> torch.Tensor:
        """Calculate integrated gradients for feature attribution."""
        self.model.eval()  # Set model to eval mode
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(X, requires_grad=True).to(self.device)
        
        # Generate scaled versions of input
        scaled_inputs = [
            baseline + (float(i) / n_steps) * (X - baseline)
            for i in range(n_steps + 1)
        ]
        
        # Compute gradients
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)  # Enable gradients
            output = self.model(scaled_input)
            
            # Ensure output requires grad
            if not output.requires_grad:
                output.requires_grad_(True)
            
            # Calculate gradients
            grad = torch.autograd.grad(
                outputs=output.sum(),
                inputs=scaled_input,
                grad_outputs=torch.ones_like(output.sum()),
                retain_graph=True,
                create_graph=True
            )[0]
            gradients.append(grad.detach())  # Detach to prevent memory leak
        
        # Stack and average gradients
        avg_grads = torch.stack(gradients).mean(dim=0)
        
        # Calculate final attributions
        attributions = (X - baseline).detach() * avg_grads
        return attributions

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Calculate feature importance using integrated gradients."""
        if not self.feature_names:
            return None
            
        # Ensure we have training data
        if not hasattr(self, '_last_X') or self._last_X is None:
            logger.warning(
                "No training data available for feature importance. "
                "Run fit() first."
            )
            return None
        
        # Calculate attributions
        try:
            with torch.set_grad_enabled(True):  # Explicitly enable gradients
                attributions = self._integrated_gradients(self._last_X)
                
                # Average absolute attributions across samples
                importance_scores = attributions.abs().mean(dim=0).cpu().numpy()
                
                # Normalize to sum to 1
                importance_scores = importance_scores / importance_scores.sum()
                
                return dict(zip(self.feature_names, importance_scores))
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit neural network."""
        # Convert data to tensors
        X_tensor = self._to_tensor(X)
        self._last_X = X_tensor  # Save for feature importance
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        n_epochs = 50
        batch_size = 32
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{n_epochs}, "
                    f"Loss: {total_loss / len(loader):.4f}"
                )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            outputs = self.model(X_tensor).squeeze()
            # Round to nearest integer and clip to valid range
            predictions = torch.round(outputs).clip(
                min=0,
                max=len(np.unique(X[self.target_col])) - 1
            )
            return predictions.cpu().numpy().astype(int)


class ModelTrainer:
    """Trains and evaluates multiple models for comparison."""
    
    def __init__(
        self,
        models: List[OrdinalModel],
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """Initialize model trainer.
        
        Args:
            models: List of models to train and evaluate
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.models = models
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def evaluate_models(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[ModelResults]:
        """Evaluate all models using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            List of ModelResults with evaluation metrics
        """
        
        results = []
        
        kf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        for model in self.models:
            logger.info(f"Evaluating {model.__class__.__name__}...")
            cv_scores = {
                'test_mae': [],
                'test_accuracy': [],
                'test_f1_weighted': [],
                'train_mae': [],
                'train_accuracy': [],
                'train_f1_weighted': []
            }
            
            # Perform cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                logger.info(f"Fold {fold_idx + 1}/{self.cv_folds}")
                
                # Split data
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                cv_scores['train_mae'].append(
                    mean_absolute_error(y_train, y_pred_train)
                )
                cv_scores['train_accuracy'].append(
                    accuracy_score(y_train, y_pred_train)
                )
                cv_scores['train_f1_weighted'].append(
                    f1_score(
                        y_train,
                        y_pred_train,
                        average='weighted'
                    )
                )
                
                cv_scores['test_mae'].append(
                    mean_absolute_error(y_test, y_pred_test)
                )
                cv_scores['test_accuracy'].append(
                    accuracy_score(y_test, y_pred_test)
                )
                cv_scores['test_f1_weighted'].append(
                    f1_score(
                        y_test,
                        y_pred_test,
                        average='weighted'
                    )
                )
            
            # Get feature importance if available
            try:
                feature_importance = model.get_feature_importance()
            except (AttributeError, ValueError):
                feature_importance = None
            
            results.append(ModelResults(
                predictions=np.concatenate([
                    y_pred_train,
                    y_pred_test
                ]),
                feature_importance=feature_importance
            ))
            
            logger.info(
                f"Average MAE: "
                f"{np.mean(cv_scores['test_mae']):.3f} ± "
                f"{np.std(cv_scores['test_mae']):.3f}"
            )
        
        return results
