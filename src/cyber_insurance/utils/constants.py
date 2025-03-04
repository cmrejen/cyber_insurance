"""Constants and enums for data processing."""
from enum import Enum
from typing import Dict, List
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
    
    # Base paths
    BASE_DIR = Path("outputs")
    MISSING_ANALYSIS_DIR = BASE_DIR / "missing_analysis"
    
    # Missing analysis subdirectories
    MISSING_GENERAL = MISSING_ANALYSIS_DIR / "general"
    MISSING_TEMPORAL = MISSING_ANALYSIS_DIR / "temporal"
    MISSING_CATEGORICAL = MISSING_ANALYSIS_DIR / "categorical"
    MISSING_CONTINUOUS = MISSING_ANALYSIS_DIR / "continuous"
    IMPUTATION = MISSING_ANALYSIS_DIR / "imputation"
    
    # Ensure all directories exist
    ALL_DIRS = [
        BASE_DIR,
        MISSING_ANALYSIS_DIR,
        MISSING_GENERAL,
        MISSING_TEMPORAL,
        MISSING_CATEGORICAL,
        MISSING_CONTINUOUS,
        IMPUTATION
    ]
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all output directories if they don't exist."""
        for directory in cls.ALL_DIRS:
            directory.mkdir(parents=True, exist_ok=True)