"""
Classical actuarial models for cyber event frequency modeling.

This module implements various zero-truncated count regression models commonly used in actuarial science.
All models are zero-truncated since we only observe companies that have experienced at least one cyber event.
The implemented models include:
- Zero-truncated Poisson: For equidispersed count data
- Zero-truncated quasi-Poisson: For under/overdispersed count data
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special
from statsmodels.discrete.truncated_model import TruncatedLFPoisson
from statsmodels.genmod.families import Poisson

from cyber_insurance.data.columns import DataColumns

logger = logging.getLogger(__name__)

class ZeroTruncatedPoissonModel:
    """Zero-truncated Poisson model using statsmodels GLM as base."""

    def __init__(self, endog, exog, offset=None):
        self.endog = endog
        self.exog = exog
        self.offset = offset if offset is not None else np.zeros(len(endog))
        self.nobs = len(endog)

        # Initialize base GLM
        self.base_model = sm.GLM(endog, exog, family=Poisson(), offset=offset)

    def loglike(self, params):
        """Log-likelihood for zero-truncated Poisson."""
        mu = np.exp(np.dot(self.exog, params) + self.offset)
        ll = (self.endog * np.log(mu) - mu - np.log(1 - np.exp(-mu)) -
              special.gammaln(self.endog + 1))
        return np.sum(ll)

    def fit(self, start_params=None, method='newton', maxiter=100):
        """Fit the model using maximum likelihood."""
        if start_params is None:
            # Get starting values from regular Poisson
            start_params = self.base_model.fit().params

        # Use statsmodels for optimization
        result = sm.GLM(
            self.endog,
            self.exog,
            family=Poisson(),
            offset=self.offset
        ).fit_constrained(
            constraints=lambda p: -self.loglike(p),
            start_params=start_params,
            method=method,
            maxiter=maxiter
        )

        return ZeroTruncatedPoissonResults(self, result)

class ZeroTruncatedPoissonResults:
    """Results class for zero-truncated Poisson model."""

    def __init__(self, model, glm_results):
        self.model = model
        self.glm_results = glm_results
        self.params = glm_results.params
        self.bse = glm_results.bse
        self.tvalues = glm_results.tvalues
        self.pvalues = glm_results.pvalues
        self.df_resid = glm_results.df_resid
        self.df_model = glm_results.df_model
        self.nobs = model.nobs

        # Calculate log-likelihood and information criteria
        self.llf = model.loglike(self.params)
        self.aic = -2 * self.llf + 2 * len(self.params)
        self.bic = -2 * self.llf + np.log(self.nobs) * len(self.params)

        # Calculate Pearson residuals for dispersion estimation
        mu = self.predict()
        self.resid_pearson = (model.endog - mu) / np.sqrt(mu)
        self.pearson_chi2 = np.sum(self.resid_pearson**2)
        self.dispersion = self.pearson_chi2 / self.df_resid

    def predict(self, exog=None, offset=None):
        """Predict expected counts."""
        if exog is None:
            exog = self.model.exog
        if offset is None:
            offset = self.model.offset

        mu = np.exp(np.dot(exog, self.params) + offset)
        # Adjust for zero truncation
        return mu / (1 - np.exp(-mu))

    def summary(self):
        """Generate summary similar to statsmodels."""
        return self.glm_results.summary()

class ClassicalFrequencyModels:
    """Classical frequency models for insurance claims modeling."""

    def __init__(self, data: pd.DataFrame, target: str = DataColumns.EVENT_FREQUENCY):
        """Initialize the models dictionary to store fitted models."""
        self.data = data
        self.target = target
        self.models = {}
        self.results = {}

    def prepare_features(self, feature_cols: Optional[list] = None):
        """Prepare features for modeling."""
        if feature_cols is None:
            feature_cols = [col for col in self.data.columns if col != self.target]

        X = self.data[feature_cols]
        y = self.data[self.target]

        return X, y

    def fit_models(self, X: np.ndarray, y: np.ndarray, exposure: Optional[np.ndarray] = None):
        """Fit all classical frequency models.

        Args:
            X: Feature matrix
            y: Target variable (number of events)
            exposure: Optional exposure variable
        """
        # Add constant term if not present
        X = sm.add_constant(X)

        # First fit zero-truncated Poisson
        self.fit_zero_truncated_poisson(X, y, exposure)

        # Calculate dispersion using Pearson residuals
        dispersion = np.sum(np.square(self.results['zero_truncated_poisson'].resid_pearson)) / self.results['zero_truncated_poisson'].df_resid

        # Add diagnostics
        logger.info("\nDispersion Analysis:")
        logger.info(f"Dispersion statistic: {dispersion:.3f}")
        logger.info(f"Sample mean of y: {np.mean(y):.3f}")
        logger.info(f"Sample variance of y: {np.var(y):.3f}")
        logger.info(f"Mean of predicted values: {np.mean(self.results['zero_truncated_poisson'].predict(X, exposure=exposure)):.3f}")
        logger.info(f"Variance of Pearson residuals: {np.var(self.results['zero_truncated_poisson'].resid_pearson):.3f}")
        logger.info(f"Degrees of freedom: {self.results['zero_truncated_poisson'].df_resid}")

        if dispersion > 1.05 or dispersion < 0.95:  # Over/underdispersion
            logger.info(f"Detected {'over' if dispersion > 1 else 'under'}dispersion")
            logger.info("Fitting zero-truncated quasi-Poisson model")
            self.fit_zero_truncated_quasi_poisson(X, y, exposure, dispersion)
        else:
            logger.info("No significant dispersion detected")

    def fit_zero_truncated_poisson(self, X: np.ndarray, y: np.ndarray,
                                 exposure: Optional[np.ndarray] = None):
        """Fit zero-truncated Poisson regression model.

        Args:
            X: Feature matrix
            y: Target variable
            exposure: Optional exposure variable
        """
        logger.info("Fitting zero-truncated Poisson regression model...")

        try:
            # Validate and clean exposure values before passing to statsmodels
            if exposure is not None:
                exposure = np.asarray(exposure, dtype=np.float64)

                # Log exposure distribution before cleaning
                logger.info("\nExposure Distribution before cleaning:")
                logger.info(f"  Min: {np.min(exposure):.4f}")
                logger.info(f"  Max: {np.max(exposure):.4f}")
                logger.info(f"  NaN count: {np.sum(np.isnan(exposure))}")
                logger.info(f"  Zero count: {np.sum(exposure == 0)}")
                logger.info(f"  Negative count: {np.sum(exposure < 0)}")

                # Replace invalid values with minimum exposure
                min_exposure = 1.0 / 12.0  # 1 month minimum
                invalid_mask = ~np.isfinite(exposure) | (exposure <= 0)
                if np.any(invalid_mask):
                    logger.warning(f"Found {np.sum(invalid_mask)} invalid exposure values. Replacing with minimum exposure (1 month)")
                    exposure = np.where(invalid_mask, min_exposure, exposure)

                # Convert exposure to log-offset
                offset = np.log(exposure)
                logger.info("\nOffset Distribution:")
                logger.info(f"  Min: {np.min(offset):.4f}")
                logger.info(f"  Max: {np.max(offset):.4f}")
            else:
                offset = None

            # Create and fit model using statsmodels' truncated Poisson with offset
            model = TruncatedLFPoisson(y, X, truncation=0, offset=offset)

            # Fit with better optimization settings
            result = model.fit(
                method='bfgs',  # BFGS typically works well for this type of model
                maxiter=1000,   # Increase max iterations
                disp=1,         # Show convergence messages
                gtol=1e-4,      # Slightly relax gradient tolerance
                full_output=1   # Get detailed optimization output
            )

            # Store model and results
            self.models['zero_truncated_poisson'] = model
            self.results['zero_truncated_poisson'] = result

            # Log model diagnostics
            logger.info("\nZero-Truncated Poisson Model Summary:")
            logger.info(result.summary().as_text())

        except Exception as e:
            logger.error(f"Failed to fit zero-truncated Poisson model: {str(e)}")
            raise

    # TODO: Implementation still needs to be completed for the quasi-Poisson model
    def fit_zero_truncated_quasi_poisson(self, X: np.ndarray, y: np.ndarray,
                                       exposure: Optional[np.ndarray] = None,
                                       dispersion: float = None):
        """Fit zero-truncated quasi-Poisson regression model.

        This is implemented by fitting a regular zero-truncated Poisson model
        and then adjusting the standard errors by multiplying them by sqrt(dispersion).
        The coefficients remain the same as the Poisson model.

        Args:
            X: Feature matrix
            y: Target variable
            exposure: Optional exposure variable
            dispersion: Dispersion parameter to adjust standard errors
        """
        logger.info("Fitting zero-truncated quasi-Poisson regression model...")

        try:
            # Validate and clean exposure values before passing to statsmodels
            if exposure is not None:
                exposure = np.asarray(exposure, dtype=np.float64)

                # Log exposure distribution before cleaning
                logger.info("\nExposure Distribution before cleaning:")
                logger.info(f"  Min: {np.min(exposure):.4f}")
                logger.info(f"  Max: {np.max(exposure):.4f}")
                logger.info(f"  NaN count: {np.sum(np.isnan(exposure))}")
                logger.info(f"  Zero count: {np.sum(exposure == 0)}")
                logger.info(f"  Negative count: {np.sum(exposure < 0)}")

                # Replace invalid values with minimum exposure
                min_exposure = 1.0 / 12.0  # 1 month minimum
                invalid_mask = ~np.isfinite(exposure) | (exposure <= 0)
                if np.any(invalid_mask):
                    logger.warning(f"Found {np.sum(invalid_mask)} invalid exposure values. Replacing with minimum exposure (1 month)")
                    exposure = np.where(invalid_mask, min_exposure, exposure)

                # Convert exposure to log-offset
                offset = np.log(exposure)
                logger.info("\nOffset Distribution:")
                logger.info(f"  Min: {np.min(offset):.4f}")
                logger.info(f"  Max: {np.max(offset):.4f}")
            else:
                offset = None

            # Create and fit model using statsmodels' truncated Poisson with offset
            model = TruncatedLFPoisson(y, X, truncation=0, offset=offset)
            result = model.fit(
                method='bfgs',  # BFGS typically works well for this type of model
                maxiter=1000,   # Increase max iterations
                disp=1,         # Show convergence messages
                gtol=1e-4,      # Slightly relax gradient tolerance
                full_output=1   # Get detailed optimization output
            )

            # Adjust standard errors for quasi-Poisson
            # Get the original standard errors from the sqrt of the diagonal of the Hessian inverse
            if hasattr(result, 'cov_params'):
                cov_params = result.cov_params()
            else:
                # If cov_params not available, compute from Hessian
                cov_params = -np.linalg.inv(result.model.hessian(result.params))

            std_errors = np.sqrt(np.diag(cov_params))

            # Adjust standard errors by multiplying by sqrt(dispersion)
            result.bse = std_errors * np.sqrt(dispersion)

            # Store model and results
            self.models['zero_truncated_quasi_poisson'] = model
            self.results['zero_truncated_quasi_poisson'] = result

            # Log model summary
            logger.info("\nZero-Truncated Quasi-Poisson Model Summary:")
            logger.info(f"Dispersion parameter: {dispersion:.3f}")
            logger.info(result.summary().as_text())

        except Exception as e:
            logger.error(f"Failed to fit zero-truncated quasi-Poisson model: {str(e)}")
            raise

    def predict_rates(self, X: np.ndarray, exposure: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Get rate predictions from fitted models.

        Args:
            X: Feature matrix for prediction
            exposure: Optional exposure values

        Returns:
            DataFrame with predicted rates from each fitted model
        """
        # Add constant term if not present
        X = sm.add_constant(X)

        # Initialize predictions dictionary
        predictions = {}

        # Get predictions from each fitted model
        for name, result in self.results.items():
            try:
                # Get predictions (already adjusted for zero-truncation)
                pred = result.predict(X, offset=exposure)

                # If exposure provided, adjust predictions
                if exposure is not None:
                    pred = pred * exposure

                predictions[f'{name}_rate'] = pred

            except Exception as e:
                logger.warning(f"Could not get predictions for {name} model: {str(e)}")

        return pd.DataFrame(predictions)

    def print_rate_interpretation(self, predictions: pd.DataFrame):
        """Print human-readable interpretation of rate predictions."""
        for col in predictions.columns:
            model_name = col.replace('_rate', '')
            logger.info(f"\n{model_name.title()} Model Predictions:")
            logger.info(f"Mean predicted rate: {predictions[col].mean():.4f}")
            logger.info(f"Median predicted rate: {predictions[col].median():.4f}")
            logger.info(f"Rate range: [{predictions[col].min():.4f}, {predictions[col].max():.4f}]")

    def save_results(self, output_path: Path):
        """Save model comparison results to a CSV file."""
        comparison_df = self.compare_models()
        comparison_df.to_csv(output_path)
        logger.info(f"Saved model comparison results to {output_path}")

    def compare_models(self) -> pd.DataFrame:
        """Compare fitted models using various metrics."""
        metrics = []

        for name, result in self.results.items():
            try:
                metrics.append({
                    'model': name,
                    'aic': result.aic,
                    'bic': result.bic,
                    'log_likelihood': result.llf,
                    'df_resid': result.df_resid,
                    'nobs': result.nobs
                })
            except Exception as e:
                logger.warning(f"Could not get metrics for {name}: {str(e)}")

        return pd.DataFrame(metrics)
