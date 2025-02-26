"""Module for preprocessing ICO breach data."""
from typing import Optional

import pandas as pd

from cyber_insurance.data.ingestion import ColumnNames
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("ico_data_preprocessing")


class ICODataPreprocessor:
    """Class for preprocessing ICO breach data."""

    def __init__(self):
        """Initialize the preprocessor."""
        self._df: Optional[pd.DataFrame] = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the ICO breach data.

        Steps:
        1. Fix duplicate BI References
        2. Convert multiple entries to single row with counts
        3. Validate data types

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        self._df = df.copy()

        # Step 1: Fix duplicate BI References
        self._fix_duplicate_bi_references()

        # Step 2: Consolidate multiple entries
        self._consolidate_multiple_entries()

        # Step 3: Validate types
        self._validate_column_types()

        return self._df

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
            ColumnNames.TIME_TAKEN_TO_REPORT.value
        ]

        # Create composite key for time period uniqueness check
        self._df["temp_key"] = (
            self._df[ColumnNames.BI_REFERENCE.value] + '_' +
            self._df[ColumnNames.YEAR.value].astype(str) + '_' +
            self._df[ColumnNames.QUARTER.value]
        )

        # Track modifications for logging
        modifications = 0
        time_period_splits = 0
        field_value_splits = 0

        # First level: Process each BI Reference group
        for bi_id, group in self._df.groupby(ColumnNames.BI_REFERENCE.value):
            if len(group) > 1:  # If we have duplicates
                # Create a composite key for the second level split
                group['field_values'] = group.apply(
                    lambda row: '_'.join(str(row[field]) for field in key_fields),
                    axis=1
                )

                # Track the next available increment for this BI Reference
                next_increment = 1

                # Keep track of whether we've seen the first occurrence
                first_occurrence_found = False

                # Process each time period group
                for temp_key, time_group in group.groupby('temp_key'):
                    if len(time_group) > 1:
                        # Check if we need to split by field values
                        unique_field_values = time_group['field_values'].unique()

                        if len(unique_field_values) > 1:
                            # We need to split this group further
                            for i, (_, field_group) in enumerate(time_group.groupby('field_values')):
                                if not first_occurrence_found:
                                    first_occurrence_found = True
                                    continue

                                new_id = f"{bi_id}A{next_increment}"
                                self._df.loc[field_group.index, ColumnNames.BI_REFERENCE.value] = new_id
                                next_increment += 1
                                modifications += len(field_group)
                                field_value_splits += 1
                        else:
                            # Only need to split by time period
                            if not first_occurrence_found:
                                first_occurrence_found = True
                            else:
                                new_id = f"{bi_id}A{next_increment}"
                                self._df.loc[time_group.index, ColumnNames.BI_REFERENCE.value] = new_id
                                next_increment += 1
                                modifications += len(time_group)
                                time_period_splits += 1
                    else:
                        # Only need to split by time period
                        if not first_occurrence_found:
                            first_occurrence_found = True
                        else:
                            new_id = f"{bi_id}A{next_increment}"
                            self._df.loc[time_group.index, ColumnNames.BI_REFERENCE.value] = new_id
                            next_increment += 1
                            modifications += len(time_group)
                            time_period_splits += 1

        # Remove temporary keys
        self._df = self._df.drop('temp_key', axis=1)

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
        data_type_count = 'Data Type Count'
        subject_type_count = 'Data Subject Type Count'

        # Validate that other columns have indeed unique values per BI Reference
        # Note that this is a fail check that is implied from the previous step
        columns_to_check = [
            ColumnNames.YEAR.value,
            ColumnNames.QUARTER.value,
            ColumnNames.DECISION_TAKEN.value,
            ColumnNames.INCIDENT_TYPE.value,
            ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value,
            ColumnNames.SECTOR.value,
            ColumnNames.TIME_TAKEN_TO_REPORT.value
        ]

        # Check uniqueness for each column
        issues = []
        for col in columns_to_check:
            # Get groups with multiple unique values
            value_counts = self._df.groupby(ColumnNames.BI_REFERENCE.value)[col].nunique()
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
                "Found multiple values in columns that should be unique:\n" +
                "\n".join(issues)
            )

        # Group by BI Reference and calculate counts
        grouped = self._df.groupby(ColumnNames.BI_REFERENCE.value).agg({
            ColumnNames.DATA_TYPE.value: lambda x: len(x.unique()),
            ColumnNames.DATA_SUBJECT_TYPE.value: lambda x: len(x.unique()),
            ColumnNames.YEAR.value: 'first',
            ColumnNames.QUARTER.value: 'first',
            ColumnNames.DECISION_TAKEN.value: 'first',
            ColumnNames.INCIDENT_TYPE.value: 'first',
            ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value: 'first',
            ColumnNames.SECTOR.value: 'first',
            ColumnNames.TIME_TAKEN_TO_REPORT.value: 'first'
        }).reset_index()

        # Rename count columns
        grouped = grouped.rename(columns={
            ColumnNames.DATA_TYPE.value: data_type_count,
            ColumnNames.DATA_SUBJECT_TYPE.value: subject_type_count
        })

        # Ensure counts are integers
        grouped[data_type_count] = grouped[data_type_count].astype(int)
        grouped[subject_type_count] = grouped[subject_type_count].astype(int)

        logger.info(f"Consolidated {len(self._df) - len(grouped)} duplicate entries")
        self._df = grouped

    def _validate_column_types(self) -> None:
        """Validate that columns have expected types after preprocessing."""
        if self._df is None:
            raise ValueError("No data loaded.")

        # Check count columns are integers
        count_cols = ['Data Type Count', 'Data Subject Type Count']
        for col in count_cols:
            if not pd.api.types.is_integer_dtype(self._df[col]):
                raise ValueError(f"Column {col} is not integer type")

        # Log column types
        logger.info("\nColumn types after preprocessing:")
        for col in self._df.columns:
            logger.info(f"{col}: {self._df[col].dtype}")

    @property
    def data(self) -> pd.DataFrame:
        """Get the current DataFrame.

        Returns:
            Current state of the DataFrame

        Raises:
            ValueError: If data has not been loaded yet
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self._df.copy()
