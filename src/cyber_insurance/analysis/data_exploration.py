"""
Data exploration script for cyber insurance events data.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path if needed
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cyber_insurance.data.ingestion import CyberEventDataLoader
from cyber_insurance.utils.logger import setup_logger

# Set up logger
log_dir = project_root / 'logs'
logger = setup_logger(
    'data_exploration',
    log_file=log_dir / 'data_exploration.log'
)

def analyze_event_frequency(data: pd.DataFrame) -> None:
    """Analyze the distribution of event frequency."""
    freq = data['event_frequency']
    
    # Basic statistics
    stats = {
        'Mean': freq.mean(),
        'Median': freq.median(),
        'Std Dev': freq.std(),
        'Variance': freq.var(),
        'Min': freq.min(),
        'Max': freq.max(),
        'Zeros': (freq == 0).sum(),
        'Zero %': (freq == 0).mean() * 100
    }
    
    logger.info("\nEvent Frequency Distribution:")
    for metric, value in stats.items():
        logger.info(f"{metric}: {value:.2f}")
    
    # Value counts
    logger.info("\nFrequency Counts:")
    value_counts = freq.value_counts().sort_index()
    for value, count in value_counts.items():
        logger.info(f"{value} events: {count} organizations ({count/len(freq)*100:.1f}%)")
    
    # Dispersion test
    variance = freq.var()
    mean = freq.mean()
    dispersion = variance / mean
    logger.info(f"\nDispersion (Variance/Mean): {dispersion:.2f}")
    if dispersion > 1:
        logger.info("Data shows overdispersion - variance > mean")
        logger.info("This suggests Negative Binomial might be more appropriate than Poisson")

def main():
    """Main function to run data exploration."""
    # Initialize data loader
    data_loader = CyberEventDataLoader()
    
    # Load and process data
    logger.info("Loading data...")
    data_loader.load_data()
    
    # Get raw data
    raw_data = data_loader.data
    
    logger.info("\n=== Raw Data Summary ===")
    logger.info(f"Number of records: {len(raw_data)}")
    logger.info("\nColumns:")
    for col in raw_data.columns:
        logger.info(f"- {col}")
    
    logger.info("\n=== Processing Frequency Data ===")
    frequency_data = data_loader.preprocess_data()
    
    # Analyze event frequency distribution
    analyze_event_frequency(frequency_data)

    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(data=frequency_data, x='event_frequency', bins=30)
    plt.title('Distribution of Cyber Events per Company')
    plt.xlabel('Number of Events')
    plt.ylabel('Count')
    
    # Save the plot
    plots_dir = project_root / 'outputs' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / 'event_frequency_distribution.png'
    plt.savefig(plot_path)
    plt.close()

    # Display summary statistics
    logger.info("\n=== Frequency Statistics ===")
    logger.info(frequency_data['event_frequency'].describe())
    logger.info(f"\nPlot saved to: {plot_path}")

if __name__ == "__main__":
    main()
