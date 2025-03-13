"""Main entry point for cyber insurance analysis pipeline."""
from pathlib import Path
from typing import List

import pandas as pd

from cyber_insurance.data.ingestion import ICODataIngestion
from cyber_insurance.data.preprocessing import ICODataPreprocessor
from cyber_insurance.models.model_trainer import (
    ModelTrainer,
    OrdinalLogistic,
    RandomForestOrdinal,
    XGBoostOrdinal,
    OrdinalModel,
    OrdinalNeuralNet
)
from cyber_insurance.models.model_evaluator import ModelEvaluator
from cyber_insurance.utils.constants import ColumnNames, InputPaths
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("main")


def run_pipeline(
    data_path: Path,
    target_col: str = ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value
) -> None:
    """Run the complete analysis pipeline.
    
    Args:
        data_path: Path to raw data file
        target_col: Name of target column for prediction
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
    
    # Step 4: Initialize models
    logger.info("Initializing models...")
    models: List[OrdinalModel] = [
        OrdinalLogistic(target_col),
        RandomForestOrdinal(target_col),
        XGBoostOrdinal(target_col),
        OrdinalNeuralNet(target_col, input_dim=X.shape[1])
    ]
    
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
    evaluator.plot_feature_importance()
    
    # Print summary
    evaluator.print_summary()
    
    # Identify best model
    best_model = evaluator.get_best_model('mae')
    logger.info(f"Best performing model (MAE): {best_model}")


if __name__ == "__main__":
    # Set data path
    data_file = Path(InputPaths.ICO_BREACH_DATA)
    
    # Run pipeline
    run_pipeline(data_file)
