"""Main entry point for cyber insurance analysis pipeline."""
from pathlib import Path
from typing import List

import pandas as pd

from cyber_insurance.data.ingestion import ICODataIngestion
from cyber_insurance.data.preprocessing import ICODataPreprocessor
from cyber_insurance.models.model_trainer import ModelTrainer
from cyber_insurance.models.model_evaluator import ModelEvaluator
from cyber_insurance.utils.constants import ColumnNames, InputPaths
from cyber_insurance.analysis.hyperparameter_tuning_analysis import (
    OrdinalLogisticTuner, RandomForestTuner, 
    OrdinalNeuralNetTuner, CatBoostOrdinalTuner
)
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("main")


def run_pipeline(
    data_path: Path,
    target_col: str = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value,
) -> None:
    """Run the complete analysis pipeline.
    
    Args:
        data_path: Path to raw data file
        target_col: Name of target column for prediction
        calculate_shap: Whether to calculate SHAP values (computationally expensive)
    """
    # Step 1: Data Ingestion
    logger.info("Starting data ingestion...")
    ingestion = ICODataIngestion()
    df = ingestion.load_data(data_path)
    
    # Step 2: Preprocessing
    logger.info("Preprocessing data...")
    preprocessor = ICODataPreprocessor()
    processed_df = preprocessor.preprocess(df)
    
    # Step 3: Prepare features and target
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    
    # Step 4: Define model factories with best hyperparameters
    logger.info("Initializing model factories with best hyperparameters...")
    
    # Create model factories with best hyperparameters
    model_factories = [
        # OrdinalLogistic
        lambda X, y: OrdinalLogisticTuner(target_col).create_model(alpha=1),
        
        # RandomForest
        lambda X, y: RandomForestTuner(target_col).create_model(
            n_estimators=100,
            min_samples_leaf=10,
            max_features=10
        ),
        
        # CatBoost
        lambda X, y: CatBoostOrdinalTuner(target_col).create_model(
            iterations=2000,
            learning_rate=0.1,
            depth=4,
            l2_leaf_reg=6
        ),
        
        # OrdinalNeuralNet - requires X and y for proper initialization
        lambda X, y: OrdinalNeuralNetTuner(target_col).create_model(
            X=X, 
            y=y,
            hidden_layer_sizes=[64, 32],
            lr=0.001,
            batch_size=64,
            epochs=100
        )
    ]
    
    # Instantiate models with data
    models = [factory(X, y) for factory in model_factories]
    
    # Step 5: Train and evaluate models
    logger.info("Training and evaluating models...")
    trainer = ModelTrainer(models)
    results = trainer.evaluate_models(X, y)
    
    # Step 6: Generate evaluation reports
    logger.info("Generating evaluation reports...")
    evaluator = ModelEvaluator(results)
    
    # Plot comparisons
    evaluator.plot_metric_comparison('mae')
    evaluator.plot_metric_comparison('accuracy')
    evaluator.plot_metric_comparison('f1_weighted')
    evaluator.plot_metric_comparison('cem')
    evaluator.plot_metric_comparison('weighted_mae')
    evaluator.plot_feature_importance()
    
    # Generate new visualizations
    logger.info("Generating advanced visualizations...")
    evaluator.plot_metrics_bar()  # Bar plots of key metrics
    evaluator.plot_error_distribution()
    
    # Print summary
    evaluator.print_summary()
    
    # Identify best model
    best_model = evaluator.get_best_model('weighted_mae')
    logger.info(f"Best performing model (Weighted MAE): {best_model}")


if __name__ == "__main__":
    # Set data path
    data_file = Path(InputPaths.ICO_BREACH_DATA)
    
    # Run pipeline
    run_pipeline(data_file)
