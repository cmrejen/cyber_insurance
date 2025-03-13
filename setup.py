"""Setup configuration for cyber insurance ML package."""
from setuptools import setup, find_packages

setup(
    name="cyber-insurance-ml",
    version="0.1.0",
    description="ML pipeline for cyber insurance frequency modeling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn==1.0.2",  # Pin version for orf compatibility
        "statsmodels>=0.14.0",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.0",
        "xgboost>=2.0.0",
        "mord>=0.7.0",
        "orf>=0.2.0",
        "setuptools==57.5.0",  # Version compatible with scikit-learn 1.0.2
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ]
    }
)
