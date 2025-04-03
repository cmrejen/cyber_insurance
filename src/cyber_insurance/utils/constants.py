"""Constants and enums for data processing."""
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


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


class DataType(str, Enum):
    """Valid data types with associated breach severity scores.
    
    Scores (1-10) are based on:
    - Financial value to attackers (ransom/sale potential)
    - Organization's willingness to pay to prevent disclosure
    - Scale of potential financial exploitation
    - Market value on dark web/to competitors
    - Potential for large-scale fraud/monetization
    """
    ECONOMIC_AND_FINANCIAL = "Economic and financial data", 10  # Highest: Direct financial exploitation, immediate monetization
    IDENTIFICATION = "Identification data", 9  # Critical: Key for identity theft, fraud at scale
    GENETIC_OR_BIOMETRIC = "Genetic or biometric data", 9  # Critical: Unique identifiers, high corporate value
    HEALTH = "Health data", 8  # High: Valuable to insurers/pharma, strong ransom leverage
    CRIMINAL_CONVICTIONS = "Criminal convictions or offences", 8  # High: Strong blackmail potential for high-value targets
    OFFICIAL_DOCUMENTS = "Official documents", 7  # Significant: Essential for identity theft schemes
    BASIC_PERSONAL_IDENTIFIERS = "Basic personal identifiers", 7  # Significant: Foundation for large-scale fraud
    SEX_LIFE = "Sex life data", 6  # Medium-high: Blackmail potential for high-net-worth individuals
    LOCATION = "Location data", 5  # Medium: Valuable for targeted attacks/stalking
    RACIAL_OR_ETHNIC = "Data revealing racial or ethnic origin", 4  # Lower: Limited direct monetization
    GENDER_REASSIGNMENT = "Gender Reassignment Data", 4  # Lower: Limited financial exploitation potential
    RELIGIOUS_OR_PHILOSOPHICAL = "Religious or philosophical beliefs", 3  # Low: Minimal financial value
    SEXUAL_ORIENTATION = "Sexual orientation data", 3  # Low: Limited large-scale exploitation potential
    TRADE_UNION = "Trade union membership", 2  # Very low: Minimal financial incentive
    POLITICAL_OPINIONS = "Political opinions", 2  # Very low: Often public, limited exploitation value

    def __new__(cls, value: str, score: int) -> 'DataType':
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.score = score
        return obj

    @classmethod
    def get_score(cls, value: str) -> int:
        """Get breach severity score for a data type value."""
        try:
            return next(member.score for member in cls if member.value == value)
        except StopIteration:
            return 0  # Default score for unknown data types


class OrdinalMapping:
    """Mappings for ordinal variables.
    
    The mappings reflect the scale of change between categories:
    - NO_SUBJECTS_AFFECTED: Exponential scale of affected subjects
    - TIME_TO_REPORT: Linear scale of reporting delay
    """
    
    NO_SUBJECTS_AFFECTED: Dict[str, int] = {
        'Unknown': 0,
        '1 to 9': 1,
        '10 to 99': 2,
        '100 to 1k': 3,
        '1k to 10k': 4,
        '10k to 100k': 5,
        '100k and above': 6
    }
    
    TIME_TO_REPORT: Dict[str, int] = {
        'Less than 24 hours': 0,
        '24 hours to 72 hours': 1,
        '72 hours to 1 week': 2,
        'More than 1 week': 3
    }
    
    _EXCLUDED = {'Unknown'}
    
    @classmethod
    def filtered_names(cls, enum_dict: Dict[str, int]) -> List[str]:
        """Get filtered category names in ordinal order (post-imputation)."""
        return sorted([k for k in enum_dict if k not in cls._EXCLUDED],
                    key=lambda k: enum_dict[k])
    
    @classmethod
    def filtered_values(cls, enum_dict: Dict[str, int]) -> List[int]:
        """Get filtered ordinal values in ordinal order (post-imputation)."""
        return [enum_dict[k] for k in cls.filtered_names(enum_dict)]


class CategoricalColumns:
    """Configuration for categorical columns that need dummy encoding."""
    
    # Columns that should be dummy encoded
    DUMMY_ENCODE_COLUMNS: List[str] = [
        ColumnNames.QUARTER.value,
        ColumnNames.DECISION_TAKEN.value,
        ColumnNames.INCIDENT_TYPE.value,
        ColumnNames.SECTOR.value
    ]
    
    # Columns that should be ordinally encoded
    ORDINAL_ENCODE_COLUMNS: Dict[str, Dict[str, int]] = {
        ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value: OrdinalMapping.NO_SUBJECTS_AFFECTED,
        ColumnNames.TIME_TAKEN_TO_REPORT.value: OrdinalMapping.TIME_TO_REPORT
    }
    
    # Valid categories for each column
    VALID_CATEGORIES: Dict[str, List[str]] = {
        ColumnNames.QUARTER.value: [
            'Qtr 1', 'Qtr 2', 'Qtr 3', 'Qtr 4'
        ],
        
        ColumnNames.DECISION_TAKEN.value: [
            'Investigation Pursued', 'No Further Action', 
            'Informal Action Taken', 'Not Yet Assigned'
        ],
        
        ColumnNames.INCIDENT_TYPE.value: [
            'Unauthorised access', 'Ransomware', 'Phishing', 'Malware',
            'Brute Force', 'Other cyber incident', 'Denial of service'
        ],
        
        ColumnNames.SECTOR.value: [
            'Finance insurance and credit', 'General business', 'Health',
            'Membership association', 'Charitable and voluntary',
            'Education and childcare', 'Retail and manufacture',
            'Land or property services', 'Religious', 
            'Online Technology and Telecoms', 'Transport and leisure',
            'Local government', 'Legal', 'Social care', 'Regulators',
            'Central Government', 'Utilities', 'Media', 'Justice',
            'Marketing', 'Political', 'Unassigned', 'Unknown'
        ]
    }
    
    # Additional ordinal columns (after encoding or transformation)
    ADDITIONAL_ORDINAL_COLUMNS: List[str] = [
        'Data Type Score',
        'Data Subject Type Count',
        'Years Since Start'
    ]
    
    @classmethod
    def get_ordinal_columns(cls) -> Set[str]:
        """Get all ordinal columns after encoding."""
        return set(cls.ORDINAL_ENCODE_COLUMNS.keys()) | set(cls.ADDITIONAL_ORDINAL_COLUMNS)
    
    @classmethod
    def is_dummy_encoded(cls, col: str) -> bool:
        """Check if a column is a dummy-encoded version of original columns."""
        return any(
            col.startswith(f"{base_col}_") 
            for base_col in cls.DUMMY_ENCODE_COLUMNS
        )

    _EXCLUDED = {
        ColumnNames.SECTOR.value: {'Unknown', 'Unassigned'}
    }
    
    @classmethod
    def filtered_categories(cls, categories: List[str], column_name: str) -> List[str]:
        """Get filtered categories list (post-imputation).
        
        Args:
            categories: List of categories to filter
            column_name: Name of the column
            
        Returns:
            Filtered list of categories
        """
        excluded = cls._EXCLUDED.get(column_name, set())
        return [cat for cat in categories if cat not in excluded]


class ModelParams:
    """Hyperparameters for machine learning models."""
    
    # Random Forest tuning ranges with proper type hints
    RF_N_ESTIMATORS: List[int] = [100, 200, 500]  # Best: MAE: 200 > CEM (SMOTEN-Under.): 500
    RF_MIN_SAMPLES_LEAF: List[int] = [3, 5, 10]  # Best: MAE: 5 > CEM (SMOTEN-Under.): 3     
    RF_MAX_FEATURES: List[int] = [5, 10, 15]  # Best: MAE: 10 > CEM (SMOTEN-Under.): 5
    
    # Ordinal Logistic parameters
    OL_ALPHA: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0]  # Regularization
    
    # CatBoost parameters
    CB_ITERATIONS: List[int] = [1000, 2000] # Best: 1000
    CB_LEARNING_RATE: List[float] = [0.01, 0.1] # Best: 0.1
    CB_DEPTH: List[int] = [4, 6] # Best: 4
    CB_L2_LEAF_REG: List[int] = [3, 6] # Best: 3
    
    # PyTorchOrdinal parameters
    PTORDINAL_HIDDEN_LAYER_SIZES: List[Tuple[int, ...]] = [(64,), (128, 64)]
    PTORDINAL_LR: List[float] = [0.001, 0.01]
    PTORDINAL_BATCH_SIZE: List[int] = [32, 64]
    PTORDINAL_EPOCHS: List[int] = [50, 100]


class InputPaths:
    """Input data paths."""
    
    # Base paths
    DATA_DIR = Path("data")
    
    # Data files
    ICO_BREACH_DATA = DATA_DIR / "data-security-cyber-incidents-trends-q1-2019-to-q3-2024.csv"
    
    # Ensure all directories exist
    ALL_DIRS = [DATA_DIR]
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all input directories if they don't exist."""
        for directory in cls.ALL_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def validate_files(cls) -> None:
        """Validate that required input files exist."""
        required_files = [cls.ICO_BREACH_DATA]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required input file not found: {file_path}"
                )


class OutputPaths:
    """Output paths for analysis results."""
    
    # Base directories
    BASE_DIR = Path("outputs")
    MISSING_ANALYSIS_DIR = BASE_DIR / "missing_analysis"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_EVALUATION_DIR = MODEL_DIR / "evaluation"
    
    # Missing analysis subdirectories
    MISSING_GENERAL = MISSING_ANALYSIS_DIR / "general"
    MISSING_TEMPORAL = MISSING_ANALYSIS_DIR / "temporal"
    MISSING_CATEGORICAL = MISSING_ANALYSIS_DIR / "categorical"
    MISSING_CONTINUOUS = MISSING_ANALYSIS_DIR / "continuous"
    IMPUTATION = MISSING_ANALYSIS_DIR / "imputation"
    
    # Hyperparameter tuning subdirectories
    HYPERPARAMETER_TUNING_DIR = MODEL_DIR / "hyperparameter_tuning"
    
    # Model-specific hyperparameter tuning directories
    CATBOOST_TUNING_DIR = HYPERPARAMETER_TUNING_DIR / "catboost"
    NEURAL_NET_TUNING_DIR = HYPERPARAMETER_TUNING_DIR / "neural_net"
    RANDOM_FOREST_TUNING_DIR = HYPERPARAMETER_TUNING_DIR / "random_forest"
    ORDINAL_LOGISTIC_TUNING_DIR = HYPERPARAMETER_TUNING_DIR / "ordinal_logistic"
    
    # SMOTE directory
    IMBALANCED_DISTRIBUTION_DIR = BASE_DIR / "smote"

    # List of all directories to create
    ALL_DIRS: List[Path] = [
        BASE_DIR,
        MISSING_ANALYSIS_DIR,
        MISSING_GENERAL,
        MISSING_TEMPORAL,
        MISSING_CATEGORICAL,
        MISSING_CONTINUOUS,
        IMPUTATION,
        MODEL_DIR,
        MODEL_EVALUATION_DIR,
        HYPERPARAMETER_TUNING_DIR,
        IMBALANCED_DISTRIBUTION_DIR,
        CATBOOST_TUNING_DIR,
        NEURAL_NET_TUNING_DIR,
        RANDOM_FOREST_TUNING_DIR,
        ORDINAL_LOGISTIC_TUNING_DIR
    ]
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all output directories if they don't exist."""
        for directory in cls.ALL_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
