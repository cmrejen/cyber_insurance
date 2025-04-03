# Cyber Insurance ML Pipeline

This project implements a machine learning pipeline for modeling cyber insurance risk using ordinal classification approaches. It features advanced preprocessing techniques, including ordinal-aware SMOTE for handling class imbalance, and multiple model implementations optimized for ordinal targets.

## Project Structure

```
cyber_insurance/
├── data/                    # Raw data storage
│   └── Cyber Events Database.xlsx
├── src/                     # Source code
│   └── cyber_insurance/
│       ├── analysis/       # Analysis modules
│       │   ├── hyperparameter_tuning_analysis.py
│       │   ├── initial_data_analysis.py
│       │   └── missing_data_analysis.py
│       ├── data/           # Data processing
│       │   ├── ingestion.py
│       │   └── preprocessing.py
│       ├── models/         # ML models
│       │   ├── hyperparameter_tuning.py
│       │   ├── model_evaluator.py
│       │   └── model_trainer.py
│       ├── utils/          # Utilities
│       │   ├── constants.py
│       │   └── logger.py
│       └── main.py         # Pipeline entry point
├── pyproject.toml          # Project dependencies
└── README.md              # Project documentation
```

## Setup

1. Install uv (if not already installed):
```bash
pip install uv
```

2. Create and activate virtual environment:
```bash
uv venv
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -e .
```

## Core Components

### 1. Data Processing (`data/`)

#### Data Ingestion (`ingestion.py`)
- Loads raw data from Excel files
- Performs initial data validation
- Handles date parsing and basic cleaning

#### Data Preprocessing (`preprocessing.py`)
- **Feature Engineering**: Creates relevant features for modeling
- **Missing Value Handling**: Uses iterative imputation
- **Class Imbalance**: Implements ordinal-aware SMOTE
- **Categorical Encoding**: Preserves ordinal relationships

### 2. Models (`models/`)

#### Model Trainer (`model_trainer.py`)
Implements four model types:
1. **Ordinal Logistic Regression**
   - Uses `mord.LogisticIT`
   - Handles ordinal targets natively
   - L2 regularization with tunable alpha

2. **Random Forest Ordinal**
   - Based on Ordered Random Forests
   - Supports honest splitting
   - Automatic hyperparameter tuning

3. **XGBoost Ordinal**
   - Modified for ordinal targets
   - Feature importance tracking
   - Multi-class softmax objective

4. **Neural Network**
   - Ordinal-aware architecture
   - Dropout regularization
   - Integrated gradients for interpretability

#### Hyperparameter Tuning (`hyperparameter_tuning.py`)
- Grid search with cross-validation
- Model-specific parameter ranges
- Performance tracking and logging

#### Model Evaluator (`model_evaluator.py`)
- Calculates metrics (MAE, accuracy, F1)
- Generates comparison plots
- Feature importance visualization

### 3. Analysis (`analysis/`)

#### Hyperparameter Analysis
- Parameter sensitivity studies
- Cross-validation results
- Learning curves

#### Data Analysis
- Distribution analysis
- Missing value patterns
- Feature correlations

## Running the Pipeline

### Basic Usage
```python
from pathlib import Path
from cyber_insurance.main import run_pipeline

# Run complete pipeline
data_path = Path("data/Cyber Events Database.xlsx")
run_pipeline(data_path)
```

### Custom Model Training
```python
from cyber_insurance.models.model_trainer import RandomForestOrdinal

# Initialize with custom parameters
model = RandomForestOrdinal(
    target_col="severity",
    n_estimators=200,
    min_samples_leaf=5
)

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Key Features

1. **Ordinal-Aware Processing**
   - Preserves ordinal relationships
   - Handles class imbalance
   - Maintains data type integrity

2. **Model Comparison Framework**
   - Consistent evaluation metrics
   - Cross-validation support
   - Feature importance analysis

3. **Extensible Architecture**
   - Abstract base classes
   - Type hints throughout
   - Comprehensive logging

4. **Performance Optimization**
   - Parallel processing support
   - Memory-efficient operations
   - GPU acceleration (Neural Network)

## Output and Visualization

The pipeline generates:
1. Model performance metrics
2. Feature importance plots
3. Cross-validation results
4. Parameter sensitivity analysis
5. Distribution comparisons

## Logging

All components use structured logging:
```python
from cyber_insurance.utils.logger import setup_logger
logger = setup_logger("component_name")
```

## Contributing

1. Follow PEP 8 guidelines
2. Include type hints
3. Add comprehensive docstrings
4. Write unit tests for new features

## Dependencies

Core requirements:
- numpy
- pandas
- scikit-learn
- torch
- xgboost
- mord
- imblearn
