# Cyber Insurance ML Pipeline

This project implements a machine learning pipeline for modeling cyber insurance risk, specifically focusing on the frequency of cyber events per firm. It compares classical actuarial techniques with modern machine learning approaches.

## Project Structure

```
cyber_insurance/
├── data/                    # Raw data storage
│   └── Cyber Events Database.xlsx
├── src/                     # Source code
│   └── cyber_insurance/
│       ├── data/           # Data processing modules
│       │   └── ingestion.py
│       ├── models/         # ML models
│       └── visualization/  # Visualization utilities
├── notebooks/              # Jupyter notebooks for analysis
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

## Usage

The main components of the pipeline are:

1. Data Ingestion: Loading and preprocessing cyber event data
2. Feature Engineering: Creating relevant features for modeling
3. Model Training: Implementation of both classical actuarial and ML models
4. Model Evaluation: Comparing model performances
5. Visualization: Analyzing and visualizing results

## Data

The project uses cyber event data with the following key features:
- Company information
- Event dates
- Event types
- Other relevant attributes

The pipeline processes this data to create a frequency-based dataset where each record represents the number of cyber events per firm.
