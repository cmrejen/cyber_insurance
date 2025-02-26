"""Module for ingesting and transforming ICO breach data."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("ico_data_ingestion")


class ColumnNames(str, Enum):
    """Column names in the ICO dataset."""
    BI_REFERENCE = 'BI Reference'
    YEAR = 'Year'
    QUARTER = 'Quarter'
    DATA_SUBJECT_TYPE = 'Data Subject Type'
    DATA_TYPE = 'Data Type'
    DECISION_TAKEN = 'Decision Taken'
    INCIDENT_TYPE = 'Incident Type'
    NO_DATA_SUBJECTS_AFFECTED = 'No. Data Subjects Affected'
    SECTOR = 'Sector'
    TIME_TAKEN_TO_REPORT = 'Time Taken to Report'


class DataTypes(str, Enum):
    """Valid data types for columns."""
    STRING = 'str'
    INTEGER = 'int'
    CATEGORY = 'category'


@dataclass
class ICODataTypes:
    """Data types for ICO breach data columns."""

    column_types: Dict[ColumnNames, DataTypes] = None

    def __post_init__(self):
        """Initialize default column types if none provided."""
        if self.column_types is None:
            self.column_types = {
                ColumnNames.BI_REFERENCE: DataTypes.STRING,
                ColumnNames.YEAR: DataTypes.INTEGER,
                ColumnNames.QUARTER: DataTypes.STRING,
                ColumnNames.DATA_SUBJECT_TYPE: DataTypes.CATEGORY,
                ColumnNames.DATA_TYPE: DataTypes.CATEGORY,
                ColumnNames.DECISION_TAKEN: DataTypes.CATEGORY,
                ColumnNames.INCIDENT_TYPE: DataTypes.CATEGORY,
                ColumnNames.NO_DATA_SUBJECTS_AFFECTED: DataTypes.CATEGORY,
                ColumnNames.SECTOR: DataTypes.CATEGORY,
                ColumnNames.TIME_TAKEN_TO_REPORT: DataTypes.CATEGORY,
            }

    def set_column_type(self, column: ColumnNames, dtype: DataTypes) -> None:
        """Set column type with validation.

        Args:
            column: Column name to set type for
            dtype: Data type to set

        Raises:
            ValueError: If column or dtype is invalid
        """
        if column not in ColumnNames:
            raise ValueError(f"Invalid column: {column}")
        if dtype not in DataTypes:
            raise ValueError(f"Invalid data type: {dtype}")
        self.column_types[column] = dtype

    def as_dict(self) -> Dict[str, str]:
        """Convert data types to dictionary format for pandas."""
        return {col.value: dtype.value for col, dtype in self.column_types.items()}


class ICODataIngestion:
    """Class for ingesting and transforming ICO breach data."""

    def __init__(self, data_types: Optional[ICODataTypes] = None):
        """Initialize the ingestion class.

        Args:
            data_types: Optional custom data types for columns.
                       If None, uses default ICODataTypes.
        """
        self.data_types = data_types or ICODataTypes()
        self._df: Optional[pd.DataFrame] = None

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load ICO breach data with appropriate data types.

        Args:
            file_path: Path to the ICO breach CSV file

        Returns:
            Raw DataFrame with appropriate data types

        Raises:
            FileNotFoundError: If file_path does not exist
            pd.errors.EmptyDataError: If file is empty
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            self._df = pd.read_csv(file_path, dtype=self.data_types.as_dict())
            logger.info(f"Loaded {len(self._df)} records from {file_path}")
            return self._df
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data file: {file_path}")
            raise e

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
