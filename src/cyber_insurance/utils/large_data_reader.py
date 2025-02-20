"""
Module for efficiently reading large data files.
"""

from pathlib import Path

import pandas as pd
from tabulate import tabulate

from cyber_insurance.utils.logger import setup_logger

# Set up logger
logger = setup_logger("large_data_reader")

def read_csv_chunks(file_path: Path, nrows: int = 5, num_chunks: int = 2) -> list[pd.DataFrame]:
    """Read multiple chunks of rows from a large CSV file using lazy loading.

    Args:
        file_path: Path to the CSV file
        nrows: Number of rows per chunk (default: 5)
        num_chunks: Number of chunks to read (default: 2)

    Returns:
        List of DataFrames, each containing nrows of data
    """
    # Use chunksize for lazy loading
    csv_iterator = pd.read_csv(
        file_path,
        chunksize=nrows,  # Read in chunks
        low_memory=True,  # Use less memory
        on_bad_lines='skip'  # Skip problematic lines
    )

    # Get the requested number of chunks
    chunks = []
    for i, chunk in enumerate(csv_iterator):
        if i >= num_chunks:
            break
        chunks.append(chunk)

    return chunks

def read_ico_breach_data(file_path: Path) -> pd.DataFrame:
    """Read and process the ICO breach dataset.
    
    Args:
        file_path: Path to the ICO breach CSV file
        
    Returns:
        Processed DataFrame with appropriate data types and transformations
    """
    logger.info("Reading ICO breach data...")
    
    # Define data types for columns
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
    
    # Read the data with specified data types
    df = pd.read_csv(
        file_path,
        dtype=dtypes,
        low_memory=True,
        on_bad_lines='skip'
    )
    
    # Create temporal features
    df['Quarter_Num'] = df['Quarter'].str.extract(r'Qtr (\d+)').astype(int)
    df['Period'] = df['Year'].astype(str) + '-Q' + df['Quarter_Num'].astype(str)
    
    # Order the severity categories
    severity_order = [
        '0',
        '1 to 9',
        '10 to 99',
        '100 to 999',
        '1k to 10k',
        '10k to 100k',
        '100k to 1m',
        'More than 1m'
    ]
    df['No. Data Subjects Affected'] = pd.Categorical(
        df['No. Data Subjects Affected'],
        categories=severity_order,
        ordered=True
    )
    
    logger.info(f"Successfully loaded {len(df)} records from ICO breach data")
    return df

def main():
    """Example usage of the module."""
    # Define the path to the ICO breach CSV file
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / "data" / "data-security-cyber-incidents-trends-q1-2019-to-q3-2024.csv"

    try:
        # Read and process ICO breach data
        df = read_ico_breach_data(csv_path)
        
        # Display sample information
        logger.info("\nSample of processed data:")
        print(tabulate(df.head(), headers='keys', tablefmt='psql'))
        
        logger.info("\nData types:")
        print(df.dtypes)
        
        logger.info("\nSeverity distribution:")
        print(df['No. Data Subjects Affected'].value_counts().sort_index())
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()
