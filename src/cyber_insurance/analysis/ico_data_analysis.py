"""
Module for comprehensive analysis of ICO breach dataset.

This module performs detailed exploratory data analysis including:
1. Data quality assessment (missing values, duplicates)
2. Analysis of multiple records per incident
3. Statistical summary of each column
4. Distribution analysis
5. Relationship between features
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("ico_data_analysis")

def load_ico_data(file_path: Path) -> pd.DataFrame:
    """Load ICO breach data with appropriate data types.
    
    Args:
        file_path: Path to the ICO breach CSV file
        
    Returns:
        Raw DataFrame with appropriate data types
    """
    dtypes = {
        'BI Reference': str,
        'Year': int,
        'Quarter': str,
        'Data Subject Type': 'category',
        'Data Type': 'category',
        'Decision Taken': 'category',
        'Incident Type': 'category',
        'No. Data Subjects Affected': 'category',
        'Sector': 'category',
        'Time Taken to Report': 'category'
    }

    return pd.read_csv(file_path, dtype=dtypes)

def analyze_multiple_records(df: pd.DataFrame) -> Dict:
    """Analyze incidents with multiple records.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing analysis results
    """
    incident_counts = df['BI Reference'].value_counts()

    analysis = {
        'total_incidents': len(incident_counts),
        'incidents_with_multiple_records': sum(incident_counts > 1),
        'max_records_per_incident': incident_counts.max(),
        'avg_records_per_incident': incident_counts.mean(),
    }

    # Analyze what differs in multiple records for same incident
    multi_record_incidents = incident_counts[incident_counts > 1].index
    multi_records = df[df['BI Reference'].isin(multi_record_incidents)]

    # Check which columns vary within same incident
    varying_columns = {}
    for incident in multi_record_incidents:  # Analyze first 5 as example
        incident_data = multi_records[multi_records['BI Reference'] == incident]
        for col in df.columns:
            if len(incident_data[col].unique()) > 1:
                varying_columns[col] = varying_columns.get(col, 0) + 1

    analysis['varying_columns'] = varying_columns
    return analysis

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values in each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing value statistics
    """
    missing = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).round(2)
    })
    return missing[missing['missing_count'] > 0]

def analyze_column_distributions(df: pd.DataFrame) -> Dict:
    """Analyze the distribution of values in each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing distribution statistics for each column
    """
    distributions = {}

    for col in df.columns:
        if df[col].dtype.name == 'category':
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)

            distributions[col] = {
                'type': 'categorical',
                'unique_values': unique_count,
                'top_5_values': value_counts.head().to_dict(),
                'distribution_stats': {
                    'mode': df[col].mode()[0],
                    'entropy': -(value_counts / len(df) * np.log(value_counts / len(df))).sum()
                }
            }
        else:
            distributions[col] = {
                'type': df[col].dtype.name,
                'summary_stats': df[col].describe().to_dict() if df[col].dtype.name != 'object' else None,
                'unique_values': df[col].nunique()
            }

    return distributions

def main():
    """Run comprehensive data analysis."""
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data" / "data-security-cyber-incidents-trends-q1-2019-to-q3-2024.csv"

    logger.info("Loading data...")
    df = load_ico_data(data_path)

    # 1. Basic dataset information
    logger.info("\nDataset Overview:")
    logger.info(f"Shape: {df.shape}")
    logger.info("\nData Types:")
    print(df.dtypes)

    # 2. Analyze multiple records per incident
    logger.info("\nAnalyzing multiple records per incident...")
    multi_record_analysis = analyze_multiple_records(df)
    logger.info("\nMultiple Records Analysis:")
    for key, value in multi_record_analysis.items():
        if key != 'varying_columns':
            logger.info(f"{key}: {value}")
    logger.info("\nColumns that vary within same incident:")
    for col, count in multi_record_analysis['varying_columns'].items():
        logger.info(f"{col}: varies in {count} incidents")

    # 3. Missing values analysis
    logger.info("\nAnalyzing missing values...")
    missing_analysis = analyze_missing_values(df)
    if not missing_analysis.empty:
        logger.info("\nMissing Values Analysis:")
        print(tabulate(missing_analysis, headers='keys', tablefmt='psql'))
    else:
        logger.info("No missing values found in the dataset")

    # 4. Column distributions
    logger.info("\nAnalyzing column distributions...")
    distributions = analyze_column_distributions(df)
    for col, stats in distributions.items():
        logger.info(f"\n{col}:")
        if stats['type'] == 'categorical':
            logger.info(f"Type: {stats['type']}")
            logger.info(f"Unique values: {stats['unique_values']}")
            logger.info("Top 5 values:")
            for val, count in stats['top_5_values'].items():
                logger.info(f"  {val}: {count}")
        else:
            logger.info(f"Type: {stats['type']}")
            logger.info(f"Unique values: {stats['unique_values']}")
            if stats['summary_stats']:
                logger.info("Summary statistics:")
                for stat, value in stats['summary_stats'].items():
                    logger.info(f"  {stat}: {value}")

if __name__ == "__main__":
    main()
