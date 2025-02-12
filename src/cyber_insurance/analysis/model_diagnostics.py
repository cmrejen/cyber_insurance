"""
Script for model diagnostics and comparison of frequency models.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to Python path if needed
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cyber_insurance.data.ingestion import CyberEventDataLoader
from cyber_insurance.models.classical import ClassicalFrequencyModels
from cyber_insurance.utils.logger import setup_logger

# Set up logger
logger = setup_logger('model_diagnostics')

def plot_exposure_distribution(data: pd.DataFrame, output_dir: Path) -> None:
    """Plot distribution of exposure times."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='exposure', bins=50)
    plt.title('Distribution of Exposure Times')
    plt.xlabel('Years')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'exposure_distribution.png')
    plt.close()

def plot_annual_rate_distribution(data: pd.DataFrame, output_dir: Path) -> None:
    """Plot distribution of annual event rates."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='annual_rate', bins=50)
    plt.title('Distribution of Annual Event Rates')
    plt.xlabel('Events per Year')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'annual_rate_distribution.png')
    plt.close()

def plot_qq_residuals(model, y: np.ndarray, output_dir: Path, model_name: str) -> None:
    """Plot Q-Q plot of Pearson residuals."""
    residuals = model.resid_pearson
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot of Pearson Residuals - {model_name}')
    plt.savefig(output_dir / f'qq_plot_{model_name.lower()}.png')
    plt.close()

def compare_rates(data: pd.DataFrame, output_dir: Path) -> None:
    """Compare observed vs predicted rates."""
    plt.figure(figsize=(10, 6))
    plt.scatter(data['annual_rate'], data['predicted_rate'], alpha=0.5)
    plt.plot([0, data['annual_rate'].max()], [0, data['annual_rate'].max()], 'r--')
    plt.title('Observed vs Predicted Annual Rates')
    plt.xlabel('Observed Rate (events/year)')
    plt.ylabel('Predicted Rate (events/year)')
    plt.savefig(output_dir / 'rate_comparison.png')
    plt.close()

def check_truncated_poisson_fit(observed_counts: np.ndarray, predicted_rates: np.ndarray) -> tuple:
    """
    Check the fit of truncated Poisson distribution to observed data.
    
    Args:
        observed_counts: Actual number of attacks per company
        predicted_rates: Predicted rates from truncated Poisson model
        
    Returns:
        Figure with diagnostic plots and summary statistics DataFrame
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Observed vs Expected Frequencies
    max_count = int(max(observed_counts))
    obs_freq = pd.Series(observed_counts).value_counts().sort_index()
    
    # Calculate expected frequencies under truncated Poisson
    exp_freq = []
    for k in range(1, max_count + 1):  # Start from 1 due to truncation
        # P(X = k | X > 0) = P(X = k) / P(X > 0)
        p_k = np.mean([stats.poisson.pmf(k, rate) / (1 - np.exp(-rate)) for rate in predicted_rates])
        exp_freq.append(p_k * len(observed_counts))
    
    ax1.bar(obs_freq.index, obs_freq.values, alpha=0.5, label='Observed')
    ax1.plot(range(1, max_count + 1), exp_freq, 'r-', label='Expected')
    ax1.set_xlabel('Number of Attacks')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Observed vs Expected Frequencies')
    ax1.legend()
    
    # 2. QQ Plot
    theoretical_quantiles = stats.poisson.ppf(
        np.linspace(0.01, 0.99, len(observed_counts)), 
        np.mean(predicted_rates)
    )
    ax2.scatter(np.sort(theoretical_quantiles), np.sort(observed_counts))
    ax2.plot([0, max(observed_counts)], [0, max(observed_counts)], 'r--')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('Q-Q Plot')
    
    # 3. Predicted Rate Distribution
    sns.histplot(predicted_rates, ax=ax3, bins=30)
    ax3.set_xlabel('Predicted Rate')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Predicted Rates')
    
    # 4. Residuals
    residuals = observed_counts - predicted_rates
    sns.scatterplot(x=predicted_rates, y=residuals, ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted Rate')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot')
    
    plt.tight_layout()
    
    # Calculate summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Mean Observed Count',
            'Mean Predicted Rate',
            'Variance Observed',
            'Variance Predicted',
            'Chi-Square Statistic',
            'Chi-Square p-value'
        ],
        'Value': [
            np.mean(observed_counts),
            np.mean(predicted_rates),
            np.var(observed_counts),
            np.var(predicted_rates),
            stats.chisquare(obs_freq, exp_freq[:len(obs_freq)])[0],
            stats.chisquare(obs_freq, exp_freq[:len(obs_freq)])[1]
        ]
    })
    
    return fig, summary_stats

def plot_rate_distributions(model_results: pd.DataFrame) -> plt.Figure:
    """
    Plot distributions of conditional and unconditional rates.
    
    Args:
        model_results: DataFrame with rate predictions
        
    Returns:
        Figure with rate distribution plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Unconditional rates
    sns.histplot(model_results['rate'], ax=ax1, bins=30)
    ax1.set_title('Distribution of Unconditional Rates')
    ax1.set_xlabel('Annual Attack Rate')
    
    # Conditional rates
    sns.histplot(model_results['rate_given_attack'], ax=ax2, bins=30)
    ax2.set_title('Distribution of Conditional Rates\n(Given At Least One Attack)')
    ax2.set_xlabel('Annual Attack Rate')
    
    plt.tight_layout()
    return fig

def plot_rate_comparison(data: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    """Plot comparison of observed and predicted rates."""
    plt.figure(figsize=(10, 6))
    plt.scatter(data['event_frequency'] / data['exposure'], predictions['rate'], alpha=0.5)
    max_rate = max(data['event_frequency'].max() / data['exposure'].min(), predictions['rate'].max())
    plt.plot([0, max_rate], [0, max_rate], 'r--')
    plt.title('Observed vs Predicted Annual Rates')
    plt.xlabel('Observed Rate (events/year)')
    plt.ylabel('Predicted Rate (events/year)')
    plt.savefig(output_dir / 'rate_comparison.png')
    plt.close()

def create_diagnostic_plots(data: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive diagnostic plots for Poisson regression."""
    # Calculate rates
    observed_rate = np.array(data['event_frequency'] / data['exposure'])
    predicted_rate = np.array(predictions['rate'])
    
    # Remove any NaN values
    mask = ~(np.isnan(observed_rate) | np.isnan(predicted_rate))
    observed_rate = observed_rate[mask]
    predicted_rate = predicted_rate[mask]
    
    # 1. Rate Comparison Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(observed_rate, predicted_rate, alpha=0.5)
    max_rate = max(observed_rate.max(), predicted_rate.max())
    plt.plot([0, max_rate], [0, max_rate], 'r--', label='Perfect Prediction')
    
    plt.title('Observed vs Predicted Attack Rates')
    plt.xlabel('Observed Rate (events/year)')
    plt.ylabel('Predicted Rate (events/year)')
    plt.legend()
    plt.savefig(output_dir / 'rate_comparison.png')
    plt.close()
    
    # 2. Rate Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(observed_rate, bins=30, alpha=0.5, label='Observed', density=True)
    plt.hist(predicted_rate, bins=30, alpha=0.5, label='Predicted', density=True)
    plt.title('Distribution of Attack Rates')
    plt.xlabel('Attack Rate (events/year)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'rate_distribution.png')
    plt.close()
    
    # 3. Q-Q Plot
    plt.figure(figsize=(10, 6))
    # Sort the data
    sorted_data = np.sort(data['event_frequency'][mask])
    # Calculate theoretical quantiles from Poisson with same mean
    mean_rate = np.mean(sorted_data)
    theoretical_quantiles = stats.poisson.ppf(np.linspace(0.01, 0.99, len(sorted_data)), mean_rate)
    
    plt.scatter(theoretical_quantiles, sorted_data, alpha=0.5)
    max_val = max(sorted_data.max(), theoretical_quantiles.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Fit')
    
    plt.title('Q-Q Plot (Poisson)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.legend()
    plt.savefig(output_dir / 'qq_plot.png')
    plt.close()
    
    # 4. Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = observed_rate - predicted_rate
    plt.scatter(predicted_rate, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Rate (events/year)')
    plt.ylabel('Residuals (Observed - Predicted)')
    plt.savefig(output_dir / 'residual_plot.png')
    plt.close()
    
    # 5. Industry Effects Plot
    plt.figure(figsize=(12, 6))
    industry_effects = pd.DataFrame({
        'Industry': ['Healthcare', 'Information', 'Professional Services', 'Public Admin', 
                    'Education', 'Finance', 'Other'],
        'Effect': [0.3483, -0.6158, 0.3506, 0.1763, -0.1346, -0.2158, 0.0],
        'Significant': [True, True, True, True, True, True, False]
    })
    
    # Sort by effect size
    industry_effects = industry_effects.sort_values('Effect', ascending=True)
    
    # Create bar plot
    bars = plt.barh(y=range(len(industry_effects)), 
                    width=industry_effects['Effect'],
                    color=['#2ecc71' if x > 0 else '#e74c3c' for x in industry_effects['Effect']])
    
    # Add industry labels
    plt.yticks(range(len(industry_effects)), industry_effects['Industry'])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            x = width + 0.01
        else:
            x = width - 0.01
        plt.text(x, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', 
                va='center')
    
    plt.title('Industry Effects on Cyber Attack Rate\n(Positive = Higher Risk, Negative = Lower Risk)', 
              pad=20)
    plt.xlabel('Effect on Log Attack Rate')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.1)
    plt.grid(True, alpha=0.1)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / 'industry_effects.png')
    plt.close()
    
    # Save summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Mean Absolute Error', 'Root Mean Squared Error', 'R-squared'],
        'Value': [
            np.mean(np.abs(residuals)),
            np.sqrt(np.mean(residuals**2)),
            1 - np.sum(residuals**2) / np.sum((observed_rate - observed_rate.mean())**2)
        ]
    })
    summary_stats.to_csv(output_dir / 'model_metrics.csv', index=False)

def plot_poisson_fit(model, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray, output_dir: Path) -> None:
    """
    Create diagnostic plots for Poisson regression fit.
    
    Args:
        model: Fitted Poisson regression model
        X: Feature matrix
        y: Observed counts
        exposure: Exposure times
        output_dir: Directory to save plots
    """
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    predicted = model.predict(X)
    plt.scatter(predicted, y, alpha=0.5)
    plt.plot([0, max(predicted)], [0, max(predicted)], 'r--')
    plt.xlabel('Predicted Count')
    plt.ylabel('Actual Count')
    plt.title('Actual vs Predicted Counts')
    plt.savefig(output_dir / 'poisson_actual_vs_predicted.png')
    plt.close()
    
    # 2. Rate Plot (accounting for exposure)
    plt.figure(figsize=(10, 6))
    actual_rate = y / exposure
    predicted_rate = predicted / exposure
    plt.scatter(predicted_rate, actual_rate, alpha=0.5)
    plt.plot([0, max(predicted_rate)], [0, max(predicted_rate)], 'r--')
    plt.xlabel('Predicted Rate (events/year)')
    plt.ylabel('Actual Rate (events/year)')
    plt.title('Actual vs Predicted Rates')
    plt.savefig(output_dir / 'poisson_rate_comparison.png')
    plt.close()
    
    # 3. Residual Plot
    residuals = (y - predicted) / np.sqrt(predicted)  # Pearson residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Count')
    plt.ylabel('Pearson Residuals')
    plt.title('Residual Plot')
    plt.savefig(output_dir / 'poisson_residuals.png')
    plt.close()
    
    # 4. Feature Effects (for categorical variables)
    for col in X.select_dtypes(include=['category', 'object']).columns:
        plt.figure(figsize=(12, 6))
        
        # Calculate mean rates per category
        category_stats = pd.DataFrame({
            'category': X[col].unique(),
            'actual_mean': [y[X[col] == cat].mean() for cat in X[col].unique()],
            'predicted_mean': [predicted[X[col] == cat].mean() for cat in X[col].unique()],
            'count': [sum(X[col] == cat) for cat in X[col].unique()]
        })
        
        # Sort by actual mean and plot
        category_stats = category_stats.sort_values('actual_mean', ascending=True)
        
        x = range(len(category_stats))
        plt.plot(x, category_stats['actual_mean'], 'bo-', label='Actual')
        plt.plot(x, category_stats['predicted_mean'], 'ro-', label='Predicted')
        
        plt.xticks(x, category_stats['category'], rotation=45, ha='right')
        plt.xlabel(col)
        plt.ylabel('Mean Event Count')
        plt.title(f'Effect of {col} on Event Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'poisson_effect_{col}.png')
        plt.close()

def main():
    """Main function to run model diagnostics."""
    # Load and prepare data
    loader = CyberEventDataLoader()
    loader.load_data()
    modeling_data = loader.preprocess_data()
    
    # Create output directory
    output_dir = project_root / 'src' / 'cyber_insurance' / 'outputs' / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and fit models
    models = ClassicalFrequencyModels(modeling_data)
    
    # Create feature matrix X and target vector y
    exclude_cols = [
        'industry_code',
        'organization',
        "first_event_date",
        "last_event_date",
        'event_frequency',
        'exposure',
        'annual_rate'
    ]
    X = modeling_data.drop(columns=exclude_cols)
    y = modeling_data['event_frequency'].values
    exposure = modeling_data['exposure'].values
    
    # Fit models
    models.fit_poisson(X, y, exposure=exposure)
    
    # Create diagnostic plots
    plot_poisson_fit(models.results['poisson'], X, y, exposure, output_dir)
    
    logger.info("Diagnostic plots saved to %s", output_dir)

if __name__ == '__main__':
    main()
