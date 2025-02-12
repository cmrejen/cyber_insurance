"""
Frequency modeling analysis for cyber insurance events.
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

from cyber_insurance.data.ingestion import CyberEventDataLoader
from cyber_insurance.data.columns import DataColumns, ColumnType
from cyber_insurance.models.classical import ClassicalFrequencyModels

# Set up logging
project_root = Path(__file__).parent.parent.parent
log_dir = project_root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'frequency_modeling.log')
    ]
)

def run_frequency_analysis():
    """Run the frequency modeling analysis pipeline."""
    # Load and preprocess data
    data_loader = CyberEventDataLoader()
    data_loader.load_data()
    modeling_data = data_loader.preprocess_data()
    
    # Log feature distributions of cyber event frequency
    logger.info("\nFeature distributions of cyber event frequency:")
    for feature in DataColumns.get_modeling_features():
        matching_cols = [col for col in modeling_data.columns if col.startswith(feature + '_') or col == feature]
        if matching_cols:
            logger.info(f"\n{feature}:")
            for col in matching_cols:  # Loop over each dummy variable for a given feature
                if col == feature:  # Original (for non-dummy categorical variable)
                    value_counts = modeling_data[col].value_counts(dropna=False)
                    for val, count in value_counts.items():
                        logger.info(f"  {val}: {count}")
                else:  # Dummy column (for dummy variables)
                    # Get the category value (everything after feature_)
                    category = col[len(feature) + 1:]
                    count = modeling_data[col].sum()
                    if count > 0:  # Only show non-zero counts
                        logger.info(f"  {category}: {count}")
    
    # Prepare feature matrix X and target vector y
    y = modeling_data[DataColumns.EVENT_FREQUENCY].values.astype(np.float64)
    
    # Create feature matrix excluding target variables
    exclude_cols = [
        DataColumns.INDUSTRY_CODE,
        DataColumns.ORGANIZATION,
        "first_event_date",
        "last_event_date",
        DataColumns.EVENT_FREQUENCY,
        DataColumns.EXPOSURE,
        DataColumns.ANNUAL_RATE
    ]
    
    # Get feature matrix
    X = modeling_data.drop(columns=exclude_cols)
    
    # Convert all columns to numeric type
    X = X.astype(np.float64)  # Convert all to float64
    
    # Add constant term
    X = sm.add_constant(X)
    
    # Get exposure time for offset
    exposure = modeling_data[DataColumns.EXPOSURE].astype(np.float64).values
    
    logger.info(f"\nFeature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    logger.info("\nFeatures used in model:")
    for col in X.columns:
        logger.info(f" - {col}")
    
    # Initialize models
    models = ClassicalFrequencyModels(modeling_data)
    
    # Test for dispersion and fit appropriate models
    try:
        # This will automatically test dispersion and fit the appropriate models
        models.fit_models(X.values, y, exposure)
        
        
        # Compare models
        model_comparison = models.compare_models()
        logger.info("\nModel Comparison Results:")
        logger.info(model_comparison.to_string())
        
        # Get and interpret predictions
        predictions = models.predict_rates(X.values, exposure=exposure)
        models.print_rate_interpretation(predictions)
        
        # Save results
        output_dir = project_root / 'outputs' / 'models'
        output_dir.mkdir(parents=True, exist_ok=True)
        models.save_results(output_dir / 'classical_models_comparison.csv')
        logger.info(f"\nResults saved to: {output_dir / 'classical_models_comparison.csv'}")
        
    except Exception as e:
        logger.error(f"Error in model fitting: {str(e)}")
        raise

if __name__ == "__main__":
    run_frequency_analysis()
