"""Module for handling data imputation."""

from typing import List, Optional

import pandas as pd
from sklearn.impute import IterativeImputer

from cyber_insurance.utils.constants import ColumnNames
from cyber_insurance.utils.logger import setup_logger
from cyber_insurance.data.preprocessing import ICODataPreprocessor

logger = setup_logger("ico_data_imputation")


class ICODataImputer:
    """Imputer for ICO breach data using MICE."""

    def __init__(self) -> None:
        """Initialize the imputer."""
        self._df: Optional[pd.DataFrame] = None
        self._preprocessor = ICODataPreprocessor()
        self._imputer = IterativeImputer(
            random_state=42, max_iter=10, min_value=0, verbose=2
        )

        # Store column order for reconstruction
        self._feature_order: List[str] = []

    def impute_data(
        self,
        df: pd.DataFrame,
        target_col: str = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value,
    ) -> pd.DataFrame:
        """Impute missing values using MICE.

        Args:
            df: Input DataFrame
            target_col: Column to impute

        Returns:
            DataFrame with imputed values
        """
        self._df = df.copy()

        # Get predictors (all columns except target)
        predictor_cols = [col for col in df.columns if col != target_col]

        # Prepare data for imputation
        imputation_df = df[predictor_cols + [target_col]].copy()

        # Use preprocessor's encoding
        logger.info("Encoding variables for imputation...")
        encoded_df = self._preprocessor.preprocess(
            imputation_df,
            encode_variables=True,
            impute_missing=False,  # Avoid recursive imputation
        )

        # Store column order
        self._feature_order = encoded_df.columns.tolist()

        # Run imputation
        logger.info("Running MICE imputation...")
        imputed_data = self._imputer.fit_transform(encoded_df)

        # Convert back to DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=self._feature_order)

        # Use preprocessor to decode back to categories
        logger.info("Decoding imputed data...")
        decoded_df = self._preprocessor.decode_variables(imputed_df)

        # Log imputation statistics
        missing_before = df[target_col].isnull().sum()
        missing_after = decoded_df[target_col].isnull().sum()
        logger.info(
            f"Imputation complete. Missing values in {target_col}: "
            f"{missing_before} -> {missing_after}"
        )

        return decoded_df
