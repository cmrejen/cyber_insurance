"""
Module for preprocessing ICO breach data.

This module handles data quality issues such as:
1. Aggregating multiple records per incident
2. Handling unknown values
3. Creating consistent severity categories
4. Feature engineering for modeling
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("data_preprocessing")


class SeverityLevel(Enum):
    """Enumeration of severity levels for cyber incidents."""
    UNKNOWN = auto()
    VERY_LOW = auto()  # 1-9
    LOW = auto()       # 10-99
    MEDIUM = auto()    # 100-999
    HIGH = auto()      # 1k-10k
    VERY_HIGH = auto() # 10k+


@dataclass
class DataQualityMetrics:
    """Container for data quality metrics."""
    total_incidents: int
    incidents_with_multiple_records: int
    max_records_per_incident: int
    unknown_severity_count: int
    unknown_subject_type_count: int
    varying_fields: Dict[str, int]


def aggregate_incident_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
    """Aggregate multiple records per incident into a single record.
    
    For each incident (unique BI Reference), this function:
    1. Takes the most severe impact (max No. Data Subjects Affected)
    2. Combines multiple Data Types and Subject Types
    3. Uses the earliest report time
    4. Preserves the first occurrence of other fields
    
    Args:
        df: Input DataFrame with potentially multiple records per incident
        
    Returns:
        Tuple containing:
        - Aggregated DataFrame with one record per incident
        - Data quality metrics
    """
    # Track varying fields within incidents
    varying_fields = {}
    for col in df.columns:
        if col not in ['BI Reference', 'Year', 'Quarter']:
            varying_count = df.groupby('BI Reference')[col].nunique()
            varying_fields[col] = sum(varying_count > 1)
    
    # Create severity order for comparison
    severity_order = [
        '0', '1 to 9', '10 to 99', '100 to 999',
        '1k to 10k', '10k to 100k', '100k to 1m', 'More than 1m'
    ]
    severity_cat = CategoricalDtype(categories=severity_order, ordered=True)
    df['No. Data Subjects Affected'] = df['No. Data Subjects Affected'].astype(severity_cat)
    
    # Prepare aggregation functions
    agg_funcs = {
        'Year': 'first',
        'Quarter': 'first',
        'Data Subject Type': lambda x: '|'.join(sorted(set(x))),
        'Data Type': lambda x: '|'.join(sorted(set(x))),
        'Decision Taken': 'first',
        'Incident Type': 'first',
        'No. Data Subjects Affected': 'max',
        'Sector': 'first',
        'Time Taken to Report': 'first'
    }
    
    # Aggregate records
    df_agg = df.groupby('BI Reference').agg(agg_funcs).reset_index()
    
    # Calculate metrics
    incident_counts = df.groupby('BI Reference').size()
    metrics = DataQualityMetrics(
        total_incidents=len(df_agg),
        incidents_with_multiple_records=sum(incident_counts > 1),
        max_records_per_incident=incident_counts.max(),
        unknown_severity_count=sum(df_agg['No. Data Subjects Affected'] == 'Unknown'),
        unknown_subject_type_count=sum(
            df_agg['Data Subject Type'].str.contains('Unknown', na=False)
        ),
        varying_fields=varying_fields
    )
    
    return df_agg, metrics


def create_severity_features(
    df: pd.DataFrame,
    handle_unknown: str = 'separate'
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Create severity features for modeling.
    
    Args:
        df: Input DataFrame
        handle_unknown: How to handle unknown severity values
                      'separate': Create binary flag for unknown
                      'impute': Use other features to impute severity
                      'remove': Remove records with unknown severity
        
    Returns:
        Tuple containing:
        - DataFrame with new severity features
        - Dictionary with counts for each severity level
    """
    # Map text categories to SeverityLevel enum
    severity_map = {
        '0': SeverityLevel.VERY_LOW,
        '1 to 9': SeverityLevel.VERY_LOW,
        '10 to 99': SeverityLevel.LOW,
        '100 to 999': SeverityLevel.MEDIUM,
        '1k to 10k': SeverityLevel.HIGH,
        '10k to 100k': SeverityLevel.VERY_HIGH,
        '100k to 1m': SeverityLevel.VERY_HIGH,
        'More than 1m': SeverityLevel.VERY_HIGH,
        'Unknown': SeverityLevel.UNKNOWN
    }
    
    # Create new severity column
    df['severity_level'] = df['No. Data Subjects Affected'].map(severity_map)
    
    # Handle unknown values based on strategy
    if handle_unknown == 'separate':
        df['is_severity_unknown'] = (df['severity_level'] == SeverityLevel.UNKNOWN)
    elif handle_unknown == 'remove':
        df = df[df['severity_level'] != SeverityLevel.UNKNOWN]
    elif handle_unknown == 'impute':
        # TODO: Implement severity imputation based on other features
        pass
    
    # Get counts for each severity level
    severity_counts = df['severity_level'].value_counts().to_dict()
    
    return df, severity_counts


def main():
    """Example usage of the preprocessing module."""
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data" / "data-security-cyber-incidents-trends-q1-2019-to-q3-2024.csv"
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Aggregate records
    logger.info("Aggregating multiple records per incident...")
    df_agg, quality_metrics = aggregate_incident_records(df)
    
    # Log data quality metrics
    logger.info("\nData Quality Metrics:")
    logger.info(f"Total incidents: {quality_metrics.total_incidents}")
    logger.info(f"Incidents with multiple records: "
                f"{quality_metrics.incidents_with_multiple_records}")
    logger.info(f"Max records per incident: {quality_metrics.max_records_per_incident}")
    logger.info(f"Unknown severity count: {quality_metrics.unknown_severity_count}")
    logger.info(f"Unknown subject type count: "
                f"{quality_metrics.unknown_subject_type_count}")
    
    logger.info("\nFields that vary within incidents:")
    for field, count in quality_metrics.varying_fields.items():
        logger.info(f"{field}: varies in {count} incidents")
    
    # Create severity features
    logger.info("\nCreating severity features...")
    df_with_severity, severity_counts = create_severity_features(
        df_agg,
        handle_unknown='separate'
    )
    
    logger.info("\nSeverity level distribution:")
    for level, count in severity_counts.items():
        logger.info(f"{level.name}: {count}")


if __name__ == "__main__":
    main()
