"""Module for preprocessing ICO breach data."""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from cyber_insurance.utils.constants import (
    ColumnNames, 
    NumericalColumns,
    CategoricalColumns,
    OutputPaths,
    DataType
)
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("ico_data_preprocessing")


class CategoricalSMOTE(SMOTENC):
    """SMOTE variant for mixed numerical and categorical features.
    
    This class extends SMOTENC since we have both numerical (Data Type Score)
    and categorical features (dummy-encoded and ordinal). SMOTENC is specifically 
    designed to handle this mix of feature types.
    
    Attributes:
        dummy_features: Set of indices for dummy-encoded features
        ordinal_features: Set of indices for ordinal features
    """
    
    def __init__(
        self,
        categorical_features: List[int],
        sampling_strategy: str = "auto",
        random_state: Optional[int] = None,
        k_neighbors: int = 5
    ) -> None:
        """Initialize CategoricalSMOTE.
        
        Args:
            categorical_features: Indices of categorical features
            sampling_strategy: Sampling strategy, default="auto"
            random_state: Random seed, default=None
            k_neighbors: Number of nearest neighbors, default=5
        """
        super().__init__(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )


class OrdinalSMOTEResampler:
    """Ordinal-aware SMOTE implementation that preserves feature types."""
    
    @staticmethod
    def calculate_sampling_strategy(y: pd.Series) -> Dict[int, int]:
        """Calculate balanced sampling strategy.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary mapping class labels to target counts
        """
        class_counts = Counter(y)
        majority_count = max(class_counts.values())
        minority_count = min(class_counts.values())
        
        # Target count for minority classes (between min and max)
        target_minority = int(np.sqrt(majority_count * minority_count))
        
        return {label: max(count, target_minority) for label, count in class_counts.items()}
    
    @staticmethod
    def get_undersampling_strategy(y: pd.Series, ratio: float = 0.75) -> Dict[int, int]:
        """Calculate undersampling strategy for majority class.
        
        Args:
            y: Target variable
            ratio: Desired ratio between second largest and majority class
            
        Returns:
            Dictionary with sampling strategy for RandomUnderSampler
        """
        counts = Counter(y)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        majority_class, majority_count = sorted_counts[0]
        second_class, second_count = sorted_counts[1]
        
        # Target count for majority class based on ratio
        target_majority = int(second_count / ratio)
        return {majority_class: target_majority}
    
    @staticmethod
    def get_oversampling_strategy(y: pd.Series) -> Dict[int, int]:
        """Calculate oversampling strategy for minority classes.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary with sampling strategy for SMOTE
        """
        counts = Counter(y)
        majority_count = max(counts.values())
        
        # Target counts for specific classes as percentage of majority
        return {
            1: int(0.60 * majority_count),
            5: int(0.50 * majority_count),
            6: int(0.70 * majority_count)
        }
    
    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: int = 42,
        sampling_strategy: Union[str, Dict[int, int]] = "auto"
    ) -> None:
        """Initialize resampler.
        
        Args:
            k_neighbors: Number of nearest neighbors
            random_state: Random seed
            sampling_strategy: Either "auto" or dict with {class_label: n_samples}
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
    
    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Resample data preserving categorical and ordinal features.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Resampled X and y
        """
        # First undersample majority class
        undersampler = RandomUnderSampler(
            sampling_strategy=self.get_undersampling_strategy(y),
            random_state=self.random_state
        )
        X_under, y_under = undersampler.fit_resample(X, y)
        
        # Group dummy columns by their original feature
        dummy_groups = {}
        dummy_features = []
        ordinal_features = []
        
        for idx, col in enumerate(X_under.columns):
            is_dummy = CategoricalColumns.is_dummy_encoded(col)
            if is_dummy:
                dummy_features.append(idx)
                base_col = col.split('_')[0]
                if base_col not in dummy_groups:
                    # Get categories in order they appear in dummy columns
                    dummy_cols = sorted([c for c in X_under.columns if c.startswith(f"{base_col}_")])
                    
                    # Get all valid categories
                    valid_cats = CategoricalColumns.VALID_CATEGORIES[base_col]
                    
                    # The first category is the one that was dropped during dummy encoding
                    # It should be the one from valid_cats that's not in any dummy column name
                    dummy_cats = set(c.split('_', 1)[1] for c in dummy_cols)
                    first_cat = next(cat for cat in valid_cats if cat not in dummy_cats)
                    
                    # Build categories list:
                    # 1. Start with the dropped (first) category
                    # 2. Add remaining categories in order they appear in dummy columns
                    categories = [first_cat]
                    for c in dummy_cols:
                        cat = c.split('_', 1)[1]
                        categories.append(cat)
                    
                    dummy_groups[base_col] = {
                        'columns': [],
                        'categories': categories
                    }
                dummy_groups[base_col]['columns'].append(col)
            elif col in CategoricalColumns.get_ordinal_columns():
                ordinal_features.append(idx)
        
        # Convert dummy-encoded back to categorical
        X_under_decoded = X_under.copy()
        for base_col, group_info in dummy_groups.items():
            dummy_cols = group_info['columns']
            categories = group_info['categories']
            
            # If all dummies are 0, use first category, else use argmax category
            X_under_decoded[base_col] = pd.Categorical(
                pd.DataFrame(X_under[dummy_cols]).apply(
                    lambda row: categories[0] if row.sum() == 0 else categories[1:][row.argmax()],
                    axis=1
                ),
                categories=categories
            )
            # Drop dummy columns
            X_under_decoded = X_under_decoded.drop(columns=dummy_cols)
        
        # Create categorical-aware SMOTE with specific strategy
        categorical_features = [
            idx for idx, col in enumerate(X_under_decoded.columns)
            if col != NumericalColumns.DATA_TYPE_SCORE.value
        ]
        
        sampler = CategoricalSMOTE(
            categorical_features=categorical_features,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
            sampling_strategy=self.get_oversampling_strategy(y_under)
        )
        
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X_under_decoded, y_under)
        
        # Verify categorical values before re-encoding
        for col, group_info in dummy_groups.items():
            categories = group_info['categories']
            actual_cats = set(X_resampled[col].unique())
            expected_cats = set(categories)
            if not actual_cats.issubset(expected_cats):
                raise ValueError(
                    f"Invalid categories in {col} after resampling: {actual_cats - expected_cats}"
                )
        
        # Convert categorical back to dummy with original categories
        X_resampled_dummy = pd.get_dummies(X_resampled, columns=dummy_groups.keys(), drop_first=True)
        
        # Convert dummy columns to float32
        for col in X_resampled_dummy.columns:
            if CategoricalColumns.is_dummy_encoded(col):
                X_resampled_dummy[col] = X_resampled_dummy[col].astype('float32')
        
        # Verify dummy columns match original
        missing_cols = set(X.columns) - set(X_resampled_dummy.columns)
        if missing_cols:
            raise ValueError(f"Missing dummy columns after encoding: {missing_cols}")
                
        # Return with columns in original order
        return X_resampled_dummy[X.columns], y_resampled
    
    def _verify_features(
        self,
        X_orig: pd.DataFrame,
        X_resampled: pd.DataFrame
    ) -> None:
        """Verify that feature characteristics are preserved.
        
        Args:
            X_orig: Original features
            X_resampled: Resampled features
        """
        for col in X_orig.columns:
            if CategoricalColumns.is_dummy_encoded(col):
                # Check binary features
                unique_vals = X_resampled[col].unique()
                if not np.all(np.isin(unique_vals, [0, 1])):
                    logger.warning(
                        f"Binary feature {col} contains non-binary values: "
                        f"{unique_vals}"
                    )
            else:
                # Check ordinal features
                orig_vals = set(X_orig[col].unique())
                new_vals = set(X_resampled[col].unique())
                if not new_vals.issubset(orig_vals):
                    logger.warning(
                        f"Ordinal feature {col} contains new values: "
                        f"{new_vals - orig_vals}"
                    )


class ICODataPreprocessor:
    """Class for preprocessing ICO breach data."""

    def __init__(self):
        """Initialize the preprocessor."""
        self._df: Optional[pd.DataFrame] = None
        self.ordinal_smote = OrdinalSMOTEResampler(
            k_neighbors=5,
            random_state=42
        )

    def preprocess(
        self,
        df: pd.DataFrame,
        encode_variables: bool = True,
        impute_missing: bool = True,
        handle_imbalance: bool = False,
    ) -> pd.DataFrame:
        """Preprocess the ICO breach data.

        Steps:
        1. Fix duplicate BI References
        2. Convert multiple entries to single row with counts
        3. Validate data types
        4. Remove low percentage unknown/unassigned values
        5. Transform year to Years Since Start (for scale reduction)
        6. Impute missing values (optional)
        7. Encode categorical and ordinal variables (optional)
        8. Handle target class imbalance using OrdinalSMOTE

        Args:
            df: Input DataFrame
            encode_variables: Whether to encode categorical and ordinal variables
            impute_missing: Whether to impute missing values using MICE

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If required columns are missing or invalid values found
        """
        self._df = df.copy()

        self._validate_columns()

        # Step 1: Fix duplicate BI References
        self._fix_duplicate_bi_references()

        # Step 2: Consolidate multiple entries
        self._consolidate_multiple_entries()

        # Step 3: Validate data types and categories
        self._validate_column_types()

        # Step 4: Remove low percentage unknown/unassigned values
        self._remove_low_percentage_unknown()

        # Step 5: Transform year to Years Since Start (for scale reduction)
        self._transform_year()

        # Step 6 & 7: Impute missing values and encode categorical variables if requested
        if impute_missing:
            logger.info("Imputing missing values...")
            if encode_variables:
                self.encode_variables()
                self._df = self._impute_missing_values()
        else:
            if encode_variables:
                logger.info("Encoding categorical variables...")
                self.encode_variables()

        # Step 8: Handle target class imbalance using OrdinalSMOTE
        if handle_imbalance:
            self._handle_class_imbalance()

        return self._df

    def minimal_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal preprocessing for initial data analysis.
        
        Only performs basic validation and year transformation,
        preserving multiple records per incident for analysis.

        Args:
            df: Input DataFrame

        Returns:
            Minimally preprocessed DataFrame

        Raises:
            ValueError: If required columns are missing or invalid values found
        """
        self._df = df.copy()

        self._validate_columns()
        self._fix_duplicate_bi_references()


        return self._df

    def _impute_missing_values(self) -> pd.DataFrame:
        """Impute missing values using MICE.

        Specifically targets 'Unknown' values (mapped to 0) in the
        NO_DATA_SUBJECTS_AFFECTED column.

        Returns:
            DataFrame with imputed values
        """
        if self._df is None:
            raise ValueError("No data loaded")

        target_col = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value

        # Store original values for comparison
        original_values = self._df[target_col].copy()

        # Mark 'Unknown' values (0) as missing for imputation
        imputation_df = self._df.copy()
        unknown_mask = imputation_df[target_col] == 0
        imputation_df.loc[unknown_mask, target_col] = np.nan

        # Initialize imputer
        imputer = IterativeImputer(
            random_state=42,
            max_iter=10,
            min_value=1,  # Ensure we don't impute back to 0 (Unknown)
            verbose=2,
        )

        # Run imputation
        logger.info("Running MICE imputation...")
        imputed_data = imputer.fit_transform(imputation_df)

        # Convert back to DataFrame
        imputed_df = pd.DataFrame(
            imputed_data, index=imputation_df.index, columns=imputation_df.columns
        )

        # Round numeric predictions for the target column only
        imputed_df[target_col] = imputed_df[target_col].round().astype(int)

        # Plot distribution comparison
        self._plot_imputation_distribution(
            original_values, imputed_df[target_col], target_col
        )

        # Log imputation statistics
        unknown_before = (original_values == 0).sum()
        unknown_after = (imputed_df[target_col] == 0).sum()
        logger.info(
            f"Imputation complete. Unknown values in {target_col}: "
            f"{unknown_before} -> {unknown_after}"
        )

        # Check class imbalance
        self._check_class_imbalance(imputed_df)

        return imputed_df

    def _check_class_imbalance(self, df: pd.DataFrame) -> None:
        """Check for class imbalance in target variable.
        
        Args:
            df: DataFrame to check
        """
        target_col = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value
        dist = Counter(df[target_col]).values()
        imbalance_ratio = max(dist) / min(dist)
        
        if imbalance_ratio > 10:
            logger.warning(
                f"Class imbalance ratio of {imbalance_ratio:.2f} detected. "
                f"This may negatively impact model performance. "
                f"Consider using class weights."
            )
        elif imbalance_ratio > 3:
            logger.info(
                f"Class imbalance ratio of {imbalance_ratio:.2f} detected. "
                f"This may impact model performance."
            )

    def _plot_imputation_distribution(
        self,
        before_data: pd.Series,
        after_data: pd.Series,
        target_col: str
    ) -> None:
        """Plot distribution comparison of values before and after imputation.
        
        Args:
            before_data: Series with values before imputation
            after_data: Series with values after imputation
            target_col: Name of the target column
        """
        # Convert to numeric, ensuring categorical is handled
        before_numeric = pd.to_numeric(before_data, errors='coerce')
        after_numeric = pd.to_numeric(after_data, errors='coerce')
        
        # Create bins for each possible value [-0.5, 0.5, 1.5, ..., 6.5]
        # This ensures each bin corresponds exactly to one integer value
        bins = np.arange(-0.5, 7, 1)
        
        # Create figure with twin axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # Plot before distribution on first axis
        sns.histplot(
            data=before_numeric,
            label='Before Imputation',
            color='blue',
            alpha=0.5,
            bins=bins,
            stat='density',
            ax=ax1
        )
        ax1.set_ylabel('Density (Before)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot after distribution on second axis
        sns.histplot(
            data=after_numeric,
            label='After Imputation',
            color='red',
            alpha=0.5,
            bins=bins,
            stat='density',
            ax=ax2
        )
        ax2.set_ylabel('Density (After)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set x-ticks to show actual values
        ax1.set_xticks(range(7))
        
        plt.title(
            f'Distribution Comparison: {target_col}\n'
            f'Before vs After Imputation'
        )
        ax1.set_xlabel('Value')
        
        # Add legends for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper right'
        )
        
        # Create output directory if needed
        OutputPaths.create_directories()
        output_dir = OutputPaths.IMPUTATION
        
        # Save plot
        plt.savefig(
            output_dir / 'imputation_distribution.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        # Log basic statistics
        logger.info("\nImputation Statistics:")
        logger.info("Before Imputation:")
        logger.info(f"Total values: {len(before_numeric)}")
        logger.info(f"Unknown (0) values: {(before_numeric == 0).sum()}")
        logger.info(
            f"Mean (excluding 0): "
            f"{before_numeric[before_numeric != 0].mean():.2f}"
        )
        
        logger.info("\nAfter Imputation:")
        logger.info(f"Total values: {len(after_numeric)}")
        logger.info(f"Unknown (0) values: {(after_numeric == 0).sum()}")
        logger.info(f"Mean: {after_numeric.mean():.2f}")

    def encode_variables(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Encode categorical and ordinal variables using predefined mappings.

        This method handles both categorical (dummy encoding) and ordinal variables
        based on the definitions in CategoricalColumns. The encoding is consistent
        whether done during preprocessing or separately.

        Args:
            df: Optional DataFrame to encode. If None, uses internal DataFrame

        Returns:
            DataFrame with encoded variables

        Raises:
            ValueError: If no DataFrame is available to encode
        """
        if df is not None:
            self._df = df.copy()
        elif self._df is None:
            raise ValueError("No DataFrame available to encode")

        # 1. Dummy encode categorical variables
        dummy_df = pd.get_dummies(
            self._df[CategoricalColumns.DUMMY_ENCODE_COLUMNS],
            drop_first=True,  # Avoid perfect multicollinearity
            prefix_sep="_",
        )
        
        # Remove columns that contain only zeros (categories not present in data)
        zero_cols = dummy_df.columns[dummy_df.sum() == 0]
        if len(zero_cols) > 0:
            logger.info(
                "Removing dummy columns with all zeros (categories not in data):\n"
                + "\n".join(f"- {col}" for col in zero_cols)
            )
            dummy_df = dummy_df.drop(columns=zero_cols)

        # Drop original columns and add dummy columns
        self._df = self._df.drop(columns=CategoricalColumns.DUMMY_ENCODE_COLUMNS)
        self._df = pd.concat([self._df, dummy_df], axis=1)

        logger.info(
            "Dummy encoded columns (with drop_first=True):\n"
            + "\n".join(f"- {col}" for col in dummy_df.columns)
        )

        # 2. Ordinal encode severity and time variables
        for col, mapping in CategoricalColumns.ORDINAL_ENCODE_COLUMNS.items():
            self._df[col] = self._df[col].map(mapping)
            logger.info(
                f"Ordinal encoded {col} with {len(mapping)} levels: "
                f"{list(mapping.items())}"
            )

        return self._df

    def _validate_columns(self) -> None:
        """Validate that all required columns are present with correct values."""
        if self._df is None:
            raise ValueError("No data loaded.")

        # Check required columns
        required_cols = [col.value for col in ColumnNames]
        missing_cols = [col for col in required_cols if col not in self._df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate categories for each column
        for col, valid_cats in CategoricalColumns.VALID_CATEGORIES.items():
            invalid_values = [
                val for val in self._df[col].unique() if val not in valid_cats
            ]
            if invalid_values:
                raise ValueError(
                    f"Invalid values in {col}: {invalid_values}\n"
                    f"Valid values are: {valid_cats}"
                )

        # Validate ordinal columns
        for col, mapping in CategoricalColumns.ORDINAL_ENCODE_COLUMNS.items():
            invalid_values = [
                val for val in self._df[col].unique() if val not in mapping
            ]
            if invalid_values:
                raise ValueError(
                    f"Invalid values in {col}: {invalid_values}\n"
                    f"Valid values are: {list(mapping.keys())}"
                )

    def _fix_duplicate_bi_references(self) -> None:
        """Fix duplicate BI References by creating unique identifiers.

        Two-level splitting process:
        1. First split based on BI Reference, Year, and Quarter
        2. Then split further if other key fields differ within same time period (it is assumed distinct cyber events occured for this date)

        For each level, we keep the original reference for the first occurrence
        and append 'A1', 'A2', etc. to subsequent ones, ensuring unique IDs even
        across multiple splits.
        """
        if self._df is None:
            raise ValueError("No data loaded.")

        # Key fields that should be unique for a given BI Reference in same time period
        key_fields = [
            ColumnNames.DECISION_TAKEN.value,
            ColumnNames.INCIDENT_TYPE.value,
            ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value,
            ColumnNames.SECTOR.value,
            ColumnNames.TIME_TAKEN_TO_REPORT.value,
        ]

        # Create composite key for time period uniqueness check
        self._df["temp_key"] = (
            self._df[ColumnNames.BI_REFERENCE.value]
            + "_"
            + self._df[ColumnNames.YEAR.value].astype(str)
            + "_"
            + self._df[ColumnNames.QUARTER.value]
        )

        # Track modifications for logging
        modifications = 0
        time_period_splits = 0
        field_value_splits = 0

        # First level: Process each BI Reference group
        for bi_id, group in self._df.groupby(ColumnNames.BI_REFERENCE.value):
            if len(group) > 1:  # If we have duplicates
                # Create a composite key for the second level split
                group["field_values"] = group.apply(
                    lambda row: "_".join(str(row[field]) for field in key_fields),
                    axis=1,
                )

                # Track the next available increment for this BI Reference
                next_increment = 1

                # Keep track of whether we've seen the first occurrence
                first_occurrence_found = False

                # Process each time period group
                for temp_key, time_group in group.groupby("temp_key"):
                    if len(time_group) > 1:
                        # Check if we need to split by field values
                        unique_field_values = time_group["field_values"].unique()

                        if len(unique_field_values) > 1:
                            # We need to split this group further
                            for i, (_, field_group) in enumerate(
                                time_group.groupby("field_values")
                            ):
                                if not first_occurrence_found:
                                    first_occurrence_found = True
                                    continue

                                new_id = f"{bi_id}A{next_increment}"
                                self._df.loc[
                                    field_group.index, ColumnNames.BI_REFERENCE.value
                                ] = new_id
                                next_increment += 1
                                modifications += len(field_group)
                                field_value_splits += 1
                        else:
                            # Only need to split by time period
                            if not first_occurrence_found:
                                first_occurrence_found = True
                            else:
                                new_id = f"{bi_id}A{next_increment}"
                                self._df.loc[
                                    time_group.index, ColumnNames.BI_REFERENCE.value
                                ] = new_id
                                next_increment += 1
                                modifications += len(time_group)
                                time_period_splits += 1
                    else:
                        # Only need to split by time period
                        if not first_occurrence_found:
                            first_occurrence_found = True
                        else:
                            new_id = f"{bi_id}A{next_increment}"
                            self._df.loc[
                                time_group.index, ColumnNames.BI_REFERENCE.value
                            ] = new_id
                            next_increment += 1
                            modifications += len(time_group)
                            time_period_splits += 1

        # Remove temporary keys
        self._df = self._df.drop("temp_key", axis=1)

        logger.info(
            f"Fixed {modifications} duplicate BI References:\n"
            f"- {time_period_splits} splits due to different time periods\n"
            f"- {field_value_splits} splits due to different field values within same time period"
        )

    def _consolidate_multiple_entries(self) -> None:
        """Consolidate multiple entries per BI Reference into a single row.

        For each BI Reference:
        1. Count unique Data Types and Data Subject Types
        2. Keep first occurrence of other fields
        3. Convert counts to integers (including single entries as 1)
        """
        if self._df is None:
            raise ValueError("No data loaded.")

        # Create new column names for counts
        data_type_score = "Data Type Score"
        subject_type_count = "Data Subject Type Count"

        # Validate that other columns have indeed unique values per BI Reference
        # Note that this is a fail check that is implied from the previous step
        columns_to_check = [
            ColumnNames.YEAR.value,
            ColumnNames.QUARTER.value,
            ColumnNames.DECISION_TAKEN.value,
            ColumnNames.INCIDENT_TYPE.value,
            ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value,
            ColumnNames.SECTOR.value,
            ColumnNames.TIME_TAKEN_TO_REPORT.value,
        ]

        # Check uniqueness for each column
        issues = []
        for col in columns_to_check:
            # Get groups with multiple unique values
            value_counts = self._df.groupby(ColumnNames.BI_REFERENCE.value)[
                col
            ].nunique()
            problematic = value_counts[value_counts > 1]

            if not problematic.empty:
                # Get example of problematic entries
                example_bi = problematic.index[0]
                example_values = self._df[
                    self._df[ColumnNames.BI_REFERENCE.value] == example_bi
                ][col].unique()

                issues.append(
                    f"Column '{col}' has multiple values for some BI References.\n"
                    f"Example: BI Reference '{example_bi}' has values: {example_values}"
                )

        if issues:
            raise ValueError(
                "Found multiple values in columns that should be unique:\n"
                + "\n".join(issues)
            )

        # Group by BI Reference and calculate counts/scores
        grouped = (
            self._df.groupby(ColumnNames.BI_REFERENCE.value)
            .agg(
                {
                    ColumnNames.DATA_TYPE.value: lambda x: sum(DataType.get_score(data_type) for data_type in x.unique()),
                    ColumnNames.DATA_SUBJECT_TYPE.value: lambda x: len(x.unique()),
                    ColumnNames.YEAR.value: "first",
                    ColumnNames.QUARTER.value: "first",
                    ColumnNames.DECISION_TAKEN.value: "first",
                    ColumnNames.INCIDENT_TYPE.value: "first",
                    ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value: "first",
                    ColumnNames.SECTOR.value: "first",
                    ColumnNames.TIME_TAKEN_TO_REPORT.value: "first",
                }
            )
            .reset_index()
        )

        # Rename columns
        grouped = grouped.rename(
            columns={
                ColumnNames.DATA_TYPE.value: data_type_score,
                ColumnNames.DATA_SUBJECT_TYPE.value: subject_type_count,
            }
        )

        # Standardize Data Type Score using z-score
        score_mean = grouped[data_type_score].mean()
        score_std = grouped[data_type_score].std()
        grouped[data_type_score] = (grouped[data_type_score] - score_mean) / score_std

        # Ensure subject type count is integer
        grouped[subject_type_count] = grouped[subject_type_count].astype(int)

        logger.info(
            f"Data Type Score standardization stats:\n"
            f"- Mean: {score_mean:.2f}\n"
            f"- Std: {score_std:.2f}\n"
            f"- Range: [{grouped[data_type_score].min():.2f}, {grouped[data_type_score].max():.2f}]"
        )
        logger.info(f"Consolidated {len(self._df) - len(grouped)} duplicate entries")
        self._df = grouped.set_index(ColumnNames.BI_REFERENCE.value)

    def _validate_column_types(self) -> None:
        """Validate that columns have expected types after preprocessing."""
        if self._df is None:
            raise ValueError("No data loaded.")

        # Check Data Subject Type Count is integer
        if not pd.api.types.is_integer_dtype(self._df["Data Subject Type Count"]):
            raise ValueError("Column Data Subject Type Count is not integer type")

        # Check Data Type Score is float
        if not pd.api.types.is_float_dtype(self._df["Data Type Score"]):
            raise ValueError("Column Data Type Score is not float type")

        # Log column types
        logger.info("\nColumn types after preprocessing:")
        for col in self._df.columns:
            logger.info(f"{col}: {self._df[col].dtype}")

    def _remove_low_percentage_unknown(self, threshold: float = 0.05) -> None:
        """Remove records with low percentage unknown/unassigned values.

        For each column, if the percentage of 'Unknown' or 'Unassigned' values
        is less than the threshold, remove those records.

        Args:
            threshold: Minimum percentage (0-1) to keep unknown values
        """
        if self._df is None:
            raise ValueError("No data loaded.")

        unknown_values = ["Unknown", "Unassigned"]

        for col in self._df.columns:
            # Skip non-categorical columns
            if not isinstance(self._df[col].dtype, pd.CategoricalDtype):
                continue

            # Calculate percentage of unknown values
            unknown_mask = self._df[col].isin(unknown_values)
            unknown_pct = unknown_mask.mean()

            # Remove if percentage is below threshold
            if 0 < unknown_pct < threshold:
                initial_count = len(self._df)
                self._df = self._df[~unknown_mask]
                removed_count = initial_count - len(self._df)

                logger.info(
                    f"Removed {removed_count} records ({unknown_pct:.2%}) with "
                    f"Unknown/Unassigned values in {col}"
                )

    def _transform_year(self) -> None:
        """Transform year to Years Since Start.

        Creates a new feature 'Years Since Start' representing the number
        of years since the earliest year in the dataset. This helps reduce
        the scale of the year variable while maintaining temporal ordering.
        """
        if self._df is None:
            raise ValueError("No data loaded.")

        min_year = self._df[ColumnNames.YEAR.value].min()
        self._df["Years Since Start"] = self._df[ColumnNames.YEAR.value] - min_year

        # Drop original year column as it's now transformed
        self._df = self._df.drop(columns=[ColumnNames.YEAR.value])

        logger.info(
            f"Transformed year to Years Since Start (reference year: {min_year})"
        )

    def _handle_class_imbalance(self) -> None:
        """Handle target class imbalance using balanced sampling.
        
        For severe imbalance (ratio > 10:1), uses a balanced strategy:
        1. Keeps majority class size unchanged
        2. Increases minority classes proportionally
        3. Maintains reasonable total dataset size
        
        Raises:
            ValueError: If DataFrame is not initialized
        """
        if self._df is None:
            raise ValueError("DataFrame must be initialized before handling imbalance")
            
        target_col = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value
        
        # Check class distribution
        class_counts = Counter(self._df[target_col])
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        
        if imbalance_ratio > 3:
            logger.info(
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})"
            )
            
            # Split features and target
            X = self._df.drop(columns=[target_col])
            y = self._df[target_col]
            
            try:
                # Calculate balanced sampling strategy
                sampling_strategy = OrdinalSMOTEResampler.calculate_sampling_strategy(y)
                
                logger.info(
                    "Using balanced sampling strategy: "
                    f"{dict(sorted(sampling_strategy.items()))}"
                )
                
                # Initialize resampler with balanced strategy
                self.ordinal_smote = OrdinalSMOTEResampler(
                    k_neighbors=5,
                    random_state=42,
                    sampling_strategy=sampling_strategy
                )
                
                # Apply resampling
                X_resampled, y_resampled = self.ordinal_smote.fit_resample(X, y)
                
                # Update DataFrame
                self._df = pd.concat([X_resampled, y_resampled], axis=1)
                
                # Log new distribution
                new_counts = Counter(y_resampled)
                new_ratio = max(new_counts.values()) / min(new_counts.values())
                logger.info(
                    f"New class distribution - imbalance ratio: {new_ratio:.2f}"
                )
                
                # Plot distribution comparison
                self._plot_class_distribution(
                    class_counts,
                    new_counts,
                    target_col
                )
                
            except Exception as e:
                logger.error(f"Failed to apply resampling: {str(e)}")
                logger.warning("Proceeding with original imbalanced data")
        else:
            logger.info(
                f"Class imbalance within acceptable range "
                f"(ratio: {imbalance_ratio:.2f})"
            )

    def _plot_class_distribution(
        self,
        original_dist: Dict[int, int],
        new_dist: Dict[int, int],
        target_col: str
    ) -> None:
        """Plot class distribution before and after resampling.
        
        Creates a bar plot comparing the class distribution before and after
        applying SMOTE resampling. The plot shows the frequency of each class
        in both distributions side by side.
        
        Args:
            original_dist: Original class distribution as Counter dict
            new_dist: Resampled class distribution as Counter dict
            target_col: Name of the target column being resampled
            
        Note:
            The plot is saved to the output directory and not displayed
            directly to maintain compatibility with non-interactive
            environments.
        """
        plt.figure(figsize=(10, 6))
        
        # Get all unique classes
        classes = sorted(set(original_dist.keys()) | set(new_dist.keys()))
        x = np.arange(len(classes))
        width = 0.35
        
        # Create bars
        plt.bar(
            x - width/2,
            [original_dist.get(c, 0) for c in classes],
            width,
            label='Original'
        )
        plt.bar(
            x + width/2,
            [new_dist.get(c, 0) for c in classes],
            width,
            label='After SMOTE'
        )
        
        # Customize plot
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title(f'Class Distribution Before and After SMOTE\n{target_col}')
        plt.xticks(x, classes)
        plt.legend()
        
        # Save plot
        OutputPaths.create_directories()
        output_path = OutputPaths.IMBALANCED_DISTRIBUTION_DIR / 'smote_distribution.png'
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Class distribution plot saved to {output_path}")

    @property
    def data(self) -> pd.DataFrame:
        """Get the current DataFrame.

        Returns:
            Current state of the DataFrame

        Raises:
            ValueError: If data has not been loaded yet
        """
        if self._df is None:
            raise ValueError("No data loaded. Call preprocess() first.")
        return self._df.copy()
