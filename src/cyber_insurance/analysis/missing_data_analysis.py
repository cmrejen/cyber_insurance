"""Module for analyzing missing data patterns in ICO breach data."""
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from cyber_insurance.utils.constants import ColumnNames, OutputPaths, InputPaths
from cyber_insurance.data.ingestion import ICODataIngestion
from cyber_insurance.data.preprocessing import ICODataPreprocessor
from cyber_insurance.utils.logger import setup_logger

logger = setup_logger("missing_data_analysis")


class MissingDataAnalyzer:
    """Analyzer for missing data patterns in ICO breach data."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self._df: Optional[pd.DataFrame] = None
        self._df_encoded: Optional[pd.DataFrame] = None
        self._missing_indicators: Dict[str, pd.Series] = {}
        
        # Create output directories
        OutputPaths.create_directories()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by converting unknown values to NaN.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with unknown values converted to NaN
        """
        self._df = df.copy()
        
        # Convert Unknown/Unassigned to NaN
        unknown_mappings = {
            ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value: ['Unknown'],
        }
        
        for col, values in unknown_mappings.items():
            # Handle categorical columns properly
            if isinstance(self._df[col].dtype, pd.CategoricalDtype):
                # Convert to string first to avoid categorical issues
                self._df[col] = self._df[col].astype(str)
            
            self._df[col] = self._df[col].replace(values, np.nan)
            
            # Create missing indicator
            self._missing_indicators[col] = pd.Series(
                self._df[col].isnull().astype(int),
                name=f"{col}_missing"
            )
            
            # Log missing value counts
            missing_count = self._df[col].isnull().sum()
            total_count = len(self._df)
            missing_pct = (missing_count / total_count) * 100
            
            logger.info(
                f"Missing values in {col}: {missing_count} "
                f"({missing_pct:.2f}%)"
            )
        
        return self._df
    
    def analyze_mar_patterns(
        self,
        save_path: Optional[Path] = None,
        use_encoded: bool = False
    ) -> None:
        """Analyze Missing At Random (MAR) patterns.
        
        Two types of analysis:
        1. Using original categorical variables (default):
           - Chi-square tests for independence
           - Cramer's V for association strength
           - Missing percentage by category
           - Temporal patterns
           
        2. Using encoded variables (if use_encoded=True):
           - Correlation analysis with numeric features
           - Point-biserial correlation with dummy variables
        
        Cramer's V interpretation:
        - < 0.1: Weak association
        - 0.1-0.3: Moderate association
        - > 0.3: Strong association
        
        Args:
            save_path: Optional path to save the plots
            use_encoded: Whether to use encoded variables for analysis
        """
        if self._df is None:
            raise ValueError("No data loaded. Call prepare_data first.")
        
        if use_encoded and self._df_encoded is None:
            raise ValueError(
                "No encoded data available. Run prepare_encoded_data first."
            )
            
        for col, indicator in self._missing_indicators.items():
            logger.info(f"\nAnalyzing patterns for {col} missingness:")
            
            # Always analyze temporal patterns
            self._analyze_temporal_patterns(col, indicator, save_path)
            
            if use_encoded:
                self._analyze_encoded_patterns(col, indicator, save_path)
            else:
                # Analyze patterns across original categories
                categorical_cols = [
                    c for c in self._df.select_dtypes(include=['category']).columns
                    if c != col  # Exclude the column being analyzed
                ]
                
                for cat_col in categorical_cols:
                    self._analyze_categorical_patterns(
                        col, indicator, cat_col, save_path
                    )
    
    def prepare_encoded_data(self, preprocessor: ICODataPreprocessor) -> None:
        """Prepare encoded version of data for additional analysis.
        
        Args:
            preprocessor: Preprocessor instance with encode_variables method
        """
        if self._df is None:
            raise ValueError("No data loaded. Call prepare_data first.")
            
        # Create encoded version of the data using predefined encodings
        self._df_encoded = preprocessor.encode_variables(df=self._df)
        logger.info("Prepared encoded version of data for analysis")
    
    def _analyze_encoded_patterns(
        self,
        target_col: str,
        indicator: pd.Series,
        save_path: Optional[Path]
    ) -> None:
        """Analyze relationships using encoded variables.
        
        Note: This analysis should be interpreted carefully as dummy variables
        from the same categorical variable are not independent.
        
        Args:
            target_col: Column being analyzed
            indicator: Missingness indicator series
            save_path: Optional path to save plots
        """
        if self._df_encoded is None:
            raise ValueError("No encoded data available")
            
        # Get numeric columns (including dummies), excluding target column
        numeric_cols = self._df_encoded.select_dtypes(
            include=['int64', 'float64']
        ).columns.drop(
            [col for col in self._df_encoded.columns 
             if col.startswith(target_col) or col == indicator.name or col == ColumnNames.NO_DATA_SUBJECTS_AFFECTED.value]
        )
        
        # Calculate correlations
        correlations = pd.Series(
            {
                col: stats.pointbiserialr(
                    self._df_encoded[col],
                    indicator
                ).correlation
                for col in numeric_cols
            }
        ).sort_values(ascending=False)
        
        # Plot correlations
        plt.figure(figsize=(12, 6))
        correlations.plot(kind='bar')
        plt.title(f"Feature Correlations with {target_col} Missingness")
        plt.xlabel("Features")
        plt.ylabel("Point-Biserial Correlation")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plot_path = OutputPaths.MISSING_CONTINUOUS / f"{target_col}_encoded_correlations.png"
            plt.savefig(plot_path)
            logger.info(f"Saved encoded correlation plot to {plot_path}")
        plt.close()
        
        # Log strong correlations
        strong_correlations = correlations[abs(correlations) > 0.1].round(3)
        if not strong_correlations.empty:
            logger.info(
                f"\nStrong correlations with {target_col} missingness "
                "(using encoded variables):"
            )
            for feat, corr in strong_correlations.items():
                logger.info(f"- {feat}: {corr:+.3f}")
                
            logger.warning(
                "Note: Correlations between dummy variables from the same "
                "categorical variable should be interpreted with caution"
            )
    
    def _analyze_temporal_patterns(
        self,
        target_col: str,
        indicator: pd.Series,
        save_path: Optional[Path]
    ) -> None:
        """Analyze relationship between missingness and time.
        
        Args:
            target_col: Column being analyzed
            indicator: Missingness indicator series
            save_path: Optional path to save plots
        """
        # Add missing indicator to DataFrame for plotting
        plot_df = self._df.copy()
        plot_df[indicator.name] = indicator
        
        # Create boxplot of years_since_start by missingness
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=plot_df,
            x=indicator.name,
            y='years_since_start'
        )
        plt.title(f"Distribution of Years by {target_col} Missingness")
        plt.xlabel("Missing")
        plt.ylabel("Years Since Start")
        plt.tight_layout()
        
        if save_path:
            plot_path = OutputPaths.MISSING_TEMPORAL / f"{target_col}_temporal_pattern.png"
            plt.savefig(plot_path)
            logger.info(f"Saved temporal pattern plot to {plot_path}")
        plt.close()
        
        # Calculate and log correlation
        corr = plot_df['years_since_start'].corr(indicator)
        logger.info(
            f"Correlation between {target_col} missingness and years: "
            f"{corr:.3f}"
        )
    
    def _analyze_categorical_patterns(
        self,
        target_col: str,
        indicator: pd.Series,
        category_col: str,
        save_path: Optional[Path]
    ) -> None:
        """Analyze relationship between missingness and categorical variables.
        
        Args:
            target_col: Column being analyzed
            indicator: Missingness indicator series
            category_col: Categorical column to analyze against
            save_path: Optional path to save plots
        """
        # Calculate contingency table and chi-square
        contingency = pd.crosstab(
            self._df[category_col],
            indicator
        )
        
        # Perform chi-square test
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        
        # Calculate Cramer's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        logger.info(
            f"\nAssociation tests for {target_col} missingness vs {category_col}:"
            f"\n- Chi-square: {chi2:.2f}"
            f"\n- p-value: {p_value:.4f}"
            f"\n- Cramer's V: {cramers_v:.3f} "
            f"({'weak' if cramers_v < 0.1 else 'moderate' if cramers_v < 0.3 else 'strong'})"
        )
        
        # Calculate and sort percentages
        missing_by_cat = (
            contingency[1] / contingency.sum(axis=1) * 100
        ).sort_values(ascending=False)
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        missing_by_cat.plot(kind='bar')
        plt.title(
            f"{target_col} Missing Percentage by {category_col}\n"
            f"(Cramer's V: {cramers_v:.3f})"
        )
        plt.xlabel(category_col)
        plt.ylabel("Percentage Missing")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plot_path = OutputPaths.MISSING_CATEGORICAL / f"{target_col}_{category_col}_pattern.png"
            plt.savefig(plot_path)
            logger.info(f"Saved {category_col} pattern plot to {plot_path}")
        plt.close()
        
        # Log categories with high missingness
        high_missing = missing_by_cat[missing_by_cat > missing_by_cat.mean()]
        if not high_missing.empty:
            logger.info(
                f"\nCategories in {category_col} with above-average "
                f"{target_col} missingness:"
            )
            for cat, pct in high_missing.items():
                logger.info(f"- {cat}: {pct:.1f}%")
                
    def plot_missing_heatmap(self, save_path: Optional[Path] = None) -> None:
        """Plot heatmap of missing values to visualize patterns.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self._df is None:
            raise ValueError("No data loaded. Call prepare_data first.")
            
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            self._df.isnull(),
            cbar=False,
            cmap="crest",
            yticklabels=False
        )
        plt.title("Missing Value Patterns")
        plt.xlabel("Features")
        plt.ylabel("Observations")
        plt.tight_layout()
        
        if save_path:
            plot_path = OutputPaths.MISSING_GENERAL / "missing_heatmap.png"
            plt.savefig(plot_path)
            logger.info(f"Saved missing value heatmap to {plot_path}")
        plt.close()
    
    def plot_missing_by_year(self, save_path: Optional[Path] = None) -> None:
        """Plot missing value percentages by year.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self._df is None:
            raise ValueError("No data loaded. Call prepare_data first.")
        
        # Calculate missing percentages by year
        missing_by_year = {}
        for col in self._missing_indicators.keys():
            yearly_missing = (
                self._df.groupby('years_since_start')[col]
                .apply(lambda x: x.isnull().mean() * 100)
            )
            missing_by_year[col] = yearly_missing
        
        # Plot
        plt.figure(figsize=(10, 6))
        for col, values in missing_by_year.items():
            plt.plot(
                values.index,
                values.values,
                marker='o',
                label=col
            )
        
        plt.title("Missing Values by Year")
        plt.xlabel("Years Since Start")
        plt.ylabel("Percentage Missing")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plot_path = OutputPaths.MISSING_TEMPORAL / "missing_by_year.png"
            plt.savefig(plot_path)
            logger.info(f"Saved missing by year plot to {plot_path}")
        plt.close()


def main() -> None:
    """Run missing data analysis."""
    # Validate input files exist
    InputPaths.validate_files()
    
    # Load and preprocess data
    ingestion = ICODataIngestion()
    df = ingestion.load_data(InputPaths.ICO_BREACH_DATA)
    
    preprocessor = ICODataPreprocessor()
    df = preprocessor.preprocess(df, encode_variables=False)
    
    # Initialize analyzer
    analyzer = MissingDataAnalyzer()
    _ = analyzer.prepare_data(df)
    
    # Create output directory if needed
    output_dir = OutputPaths.MISSING_ANALYSIS_DIR
    
    # Plot missing value patterns
    logger.info("\nGenerating missing value visualizations...")
    analyzer.plot_missing_heatmap(
        output_dir / "missing_heatmap.png"
    )
    
    analyzer.plot_missing_by_year(
        output_dir / "missing_by_year.png"
    )
    
    # Analyze MAR patterns
    logger.info("\nAnalyzing MAR patterns...")
    
    # First analyze with original categorical variables
    analyzer.analyze_mar_patterns(output_dir, use_encoded=False)
    
    # Optionally analyze with encoded variables
    logger.info("\nPreparing encoded data for additional analysis...")
    analyzer.prepare_encoded_data(preprocessor)
    analyzer.analyze_mar_patterns(output_dir, use_encoded=True)
    
    logger.info("Missing data analysis complete.")


if __name__ == "__main__":
    main()
