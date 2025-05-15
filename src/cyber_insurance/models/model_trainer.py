"""Model training and evaluation framework for cyber insurance data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from mord import LogisticIT
import orf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (mean_absolute_error, accuracy_score, f1_score)
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import CornLoss, coral_loss
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import corn_label_from_logits

from cyber_insurance.utils.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    predictions: np.ndarray
    cv_scores: Dict[str, np.ndarray]
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
        inference: bool = False,
        replace: bool = True, # Bootstrapping or subsampling
        honesty: bool = False,
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
        self.replace = replace
        self.model: Optional[orf.OrderedForest] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model with hyperparameter tuning if needed."""
        try:
            self.feature_names = list(X.columns)
            
            # Initialize and train model
            self.model = orf.OrderedForest(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                replace=self.replace,
                sample_fraction=self.sample_fraction,
                honesty=self.honesty,
                honesty_fraction=self.honesty_fraction,
                inference=self.inference,
                random_state=self.random_state
            )
            
            # Convert data to numpy arrays with proper types
            X_values = np.asarray(X.values, dtype=np.float64)
            y_values = np.asarray(y.values, dtype=np.int32).reshape(-1)
            
            self.model.fit(X_values, y_values)
            
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            raise ValueError(f"Model fitting failed: {e}") from e
    
    def predict(self, X: pd.DataFrame, prob: bool = False) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")
        
        try:
            X_values = np.asarray(X.values, dtype=np.float64)
            
            # Get predictions and handle different return types
            y_pred = self.model.predict(X_values, prob=prob)["predictions"]
            
            return y_pred.reshape(-1)  # Ensure 1D array
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to generate predictions: {e}") from e

    
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


class OrdinalLogistic(OrdinalModel):
    """Ordinal regression using mord's LogisticIT model."""
    
    def __init__(self, target_col: str, alpha: float = 0.001) -> None:
        super().__init__(target_col)
        self.alpha = alpha
        self.model = LogisticIT(alpha=alpha)
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


class OrdinalDataset(Dataset):
    """Dataset class for ordinal data."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, cat_cols: List[str]) -> None:
        """Initialize dataset.
        
        Args:
            X: Feature DataFrame
            y: Target series
            cat_cols: List of categorical column names
        
        Note:
            For CORAL, target values are shifted to start from 0 if needed.
            This ensures proper handling of ordinal boundaries.
        """
        self.X = X
        
        # Ensure labels start from 0 for CORAL
        min_label = y.min()
        if min_label > 0:
            self.y = y - min_label
        else:
            self.y = y
            
        self.cat_cols = cat_cols

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index.
        
        Args:
            idx: Index to retrieve
            
        Returns:
            Dictionary with numerical features tensor and target
        """
        # Convert all features to float tensor
        x_num = torch.FloatTensor(self.X.iloc[idx].values)
        y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
        
        return {'numerical': x_num}, y


class CornNN(nn.Module):
    """Neural Network with CORN (Conditional Ordinal Neural Network) architecture."""
    
    def __init__(
        self,
        num_numerical_features: int,
        num_classes: int,
        hidden_sizes: tuple = (128, 64),
        dropout: float = 0.1
    ) -> None:
        """Initialize network.
        
        Args:
            num_numerical_features: Number of numerical features
            num_classes: Number of classes (K)
            hidden_sizes: Sizes of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_size = num_numerical_features
        
        # Hidden layers
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = size
        
        # Shared feature extractor
        self.shared = nn.Sequential(*layers)
        
        # Single output layer for K-1 binary classifiers
        self.output_layer = nn.Linear(prev_size, num_classes - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, num_features]
            
        Returns:
            Logits tensor [batch_size, K-1]
        """
        features = self.shared(x)
        return self.output_layer(features)


class OrdinalNeuralNet(OrdinalModel):
    """Neural Network for ordinal classification using CORN."""
    
    def __init__(
        self,
        target_col: str,
        num_numerical_features: int,
        num_classes: int,
        hidden_layer_sizes: tuple = (128, 64),
        lr: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        random_state: int = 42,
        dropout: float = 0.1
    ) -> None:
        """Initialize model.
        
        Args:
            target_col: Name of target column
            num_numerical_features: Number of numerical features
            num_classes: Total number of classes (K)
            hidden_layer_sizes: Sizes of hidden layers
            lr: Initial learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
            random_state: Random seed
            dropout: Dropout rate
        """
        super().__init__(target_col)
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.dropout = dropout
        
        # Initialize model
        self.model = CornNN(
            num_numerical_features=num_numerical_features,
            num_classes=num_classes,
            hidden_sizes=hidden_layer_sizes,
            dropout=dropout
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize CORN loss function
        self.criterion = CornLoss(num_classes=num_classes)
        
        # Add learning rate scheduler with optimized parameters
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.2,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model to data.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        torch.manual_seed(self.random_state)
        self.model.train()
        
        # Create dataset and dataloader, y is expected to be on a 0-based scale
        dataset = OrdinalDataset(X, y, [])
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        epoch_losses = []
        for epoch in range(self.epochs):
            batch_losses = []
            for i, (data, labels) in enumerate(dataloader):
                x_num = data['numerical'].to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                logits = self.model(x_num)
                
                # Use CORN loss
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                batch_losses.append(loss.item())
                
                if (i+1) % 10 == 0:
                    logger.info(
                        f'Epoch {epoch+1}/{self.epochs}, '
                        f'Step {i+1}, Loss: {loss.item():.4f}'
                    )
            
            # Average loss for this epoch
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            
            # Update learning rate scheduler with epoch loss
            self.scheduler.step(epoch_loss)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using CORN model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted class labels
        """
        self.model.eval()
        dataset = OrdinalDataset(X, pd.Series(np.zeros(len(X))), [])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        with torch.no_grad():
            for data, _ in dataloader:
                x_num = data['numerical'].to(device)
                logits = self.model(x_num)
                pred = corn_label_from_logits(logits)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions).numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities using CORN model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of class probabilities [batch_size, K-1]
        """
        self.model.eval()
        dataset = OrdinalDataset(X, pd.Series(np.zeros(len(X))), [])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        probabilities = []
        with torch.no_grad():
            for data, _ in dataloader:
                x_num = data['numerical'].to(device)
                logits = self.model(x_num)
                # Convert logits to probabilities and compute cumulative product
                probas = torch.sigmoid(logits)
                probas = torch.cumprod(probas, dim=1)
                probabilities.append(probas.cpu())
        
        return torch.cat(probabilities).numpy()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores using gradient-based attribution.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None if
            feature names are not available
        """
        if not hasattr(self, 'feature_names'):
            return None
            
        self.model.eval()
        importances = {}
        
        # Create a small batch of random data
        X_sample = torch.randn(100, self.num_numerical_features, device=device)
        X_sample.requires_grad_(True)
        
        # Get predictions
        logits = self.model(X_sample)
        
        # Calculate gradients
        logits.sum().backward()
        
        # Feature importance is the mean absolute gradient
        importance = X_sample.grad.abs().mean(0)
        
        # Assign to numerical features
        num_features = self.feature_names
        for feat, imp in zip(num_features, importance):
            importances[feat] = imp.item()
            
        return importances


class CatBoostOrdinal(OrdinalModel):
    """CatBoost Classifier for ordinal classification."""
    
    def __init__(
        self,
        target_col: str,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        l2_leaf_reg: float = 3,
        loss_function: str = 'MultiClass',
        eval_metric: str = 'TotalF1',
        random_state: int = 42,
        verbose: bool = False,
        auto_class_weights: str = 'Balanced'
    ):
        super().__init__(target_col)
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
            auto_class_weights=auto_class_weights
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y, verbose=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).squeeze()
    
    def get_feature_importance(self) -> Dict[str, float]:
        return dict(zip(
            self.model.feature_names_,
            self.model.get_feature_importance()
        ))


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
                X_train = X.iloc[train_idx].copy()
                y_train = y.iloc[train_idx].copy()
                X_test = X.iloc[test_idx].copy()
                y_test = y.iloc[test_idx].copy()
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics (in-sample)
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
                
                # Calculate metrics (out-of-sample)
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
                predictions=np.concatenate([
                    y_pred_train,
                    y_pred_test
                ]),
                cv_scores=cv_scores,
                feature_importance=feature_importance
            ))
            
            # Log average (out-of-sample) metrics
            logger.info(
                f"Average MAE: "
                f"{np.mean(cv_scores['test_mae']):.3f} ± "
                f"{np.std(cv_scores['test_mae']):.3f}"
            )
        
        return results
