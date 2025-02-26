"""Constants and enums for data processing."""
from enum import Enum
from typing import Dict, List


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