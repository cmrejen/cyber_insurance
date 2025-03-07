"""Model training and evaluation framework for cyber insurance data.

This module implements various models suitable for ordinal classification,
from classical approaches to modern machine learning methods.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import torch
import torch.nn as nn
from xgboost import XGBClassifier
logger = logging.getLogger(__name__)

@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    cv_scores: Dict[str, List[float]]
    feature_importance: Optional[Dict[str, float]] = None


class OrdinalModel(ABC):
    """Abstract base class for ordinal classification models."""
    
    def __init__(self, target_col: str):
        """Initialize ordinal model.
        
        Args:
            target_col: Name of target column
        """
        self.target_col = target_col
        self.model: Optional[BaseEstimator] = None
        self.label_encoder = LabelEncoder()
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return None


class RandomForestOrdinal(OrdinalModel):
    """Random Forest for ordinal classification."""
    
    def __init__(
        self,
        target_col: str,
        n_estimators: int = 100,
        max_depth: Optional[int] = None
    ):
        super().__init__(target_col)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def get_feature_importance(self) -> Dict[str, float]:
        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))


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
            objective='reg:squarederror',  # Treats as regression
            random_state=42
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(
            np.round(y_pred).astype(int)
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))


class OrdinalLogistic(OrdinalModel):
    """Ordinal Logistic Regression using statsmodels.
    
    This implementation uses statsmodels' OrderedModel which properly accounts for
    the ordinal nature of the target variable by modeling cumulative probabilities.
    """
    
    def __init__(self, target_col: str):
        """Initialize ordinal logistic model.
        
        Args:
            target_col: Name of target column
        """
        super().__init__(target_col)
        self.fitted_model = None
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit ordinal logistic model.
        
        Args:
            X: Feature DataFrame
            y: Target series with ordinal values
        """
        self.feature_names = X.columns
        y_encoded = self.label_encoder.fit_transform(y)
        
        # statsmodels requires a pandas DataFrame
        model = OrderedModel(
            y_encoded,
            X,
            distr='logit'  # Use logistic distribution
        )
        
        # Fit model with maximum likelihood estimation
        try:
            self.fitted_model = model.fit(
                method='bfgs',  # More stable than default Newton
                maxiter=1000,
                disp=False
            )
        except Exception as e:
            logger.error(f"Error fitting ordinal logistic model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ordinal classes.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted ordinal classes
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predicting")
        
        # Get predicted probabilities for each class
        probs = self.fitted_model.predict(X)
        
        # Convert to class predictions (highest probability class)
        y_pred = np.argmax(probs, axis=1)
        
        return self.label_encoder.inverse_transform(y_pred)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on coefficient magnitudes.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting importance")
            
        # Get absolute values of coefficients
        coef = np.abs(self.fitted_model.params[self.feature_names])
        
        return dict(zip(self.feature_names, coef))


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
        y_encoded = self.label_encoder.fit_transform(y)
        y_tensor = torch.FloatTensor(y_encoded).to(self.device)
        
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
                max=len(self.label_encoder.classes_) - 1
            )
            return self.label_encoder.inverse_transform(
                predictions.cpu().numpy().astype(int)
            )


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
        from sklearn.metrics import (
            mean_absolute_error,
            accuracy_score,
            f1_score
        )
        
        results = []
        kf = KFold(
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
            
            # Perform cross-validation manually
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
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
                model_name=model.__class__.__name__,
                cv_scores=cv_scores,
                feature_importance=feature_importance
            ))
            
            logger.info(
                f"Average MAE: "
                f"{np.mean(cv_scores['test_mae']):.3f} ± "
                f"{np.std(cv_scores['test_mae']):.3f}"
            )
        
        return results
