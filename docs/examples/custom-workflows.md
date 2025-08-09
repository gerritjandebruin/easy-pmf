# Custom Workflows and Advanced Techniques

This guide covers advanced PMF analysis techniques and customizations beyond the standard workflow, including specialized data handling, custom visualizations, and advanced statistical methods.

## Overview

While Easy PMF provides a streamlined interface for standard PMF analysis, real-world applications often require customizations such as:

1. Custom data preprocessing techniques
2. Specialized uncertainty handling
3. Advanced factor selection methods
4. Custom visualization and reporting
5. Integration with other analysis tools

## Advanced Data Preprocessing

### Custom Missing Data Treatment

```python
from easy_pmf import PMF
import pandas as pd
import numpy as np

class CustomPMF(PMF):
    """Extended PMF class with custom preprocessing methods."""

    def custom_missing_data_treatment(self, method='seasonal_median'):
        """Apply custom missing data treatment."""

        if method == 'seasonal_median':
            # Replace missing values with seasonal medians
            data = self.concentration_data.copy()

            # Assume we have a datetime index
            if hasattr(data.index, 'month'):
                for month in range(1, 13):
                    monthly_mask = data.index.month == month
                    monthly_data = data[monthly_mask]

                    for column in data.columns:
                        monthly_median = monthly_data[column].median()
                        data.loc[monthly_mask, column] = data.loc[monthly_mask, column].fillna(monthly_median)

            self.concentration_data = data
            print(f"Applied seasonal median imputation")

        elif method == 'interpolation':
            # Use interpolation for time series data
            self.concentration_data = self.concentration_data.interpolate(method='time')
            print(f"Applied time-based interpolation")

    def apply_log_transformation(self, species_list=None):
        """Apply log transformation to specified species."""

        if species_list is None:
            # Apply to all species with high dynamic range
            species_list = []
            for col in self.concentration_data.columns:
                data_range = self.concentration_data[col].max() / self.concentration_data[col].min()
                if data_range > 100:  # High dynamic range
                    species_list.append(col)

        for species in species_list:
            if species in self.concentration_data.columns:
                # Add small constant to avoid log(0)
                min_val = self.concentration_data[species][self.concentration_data[species] > 0].min()
                self.concentration_data[species] = np.log10(
                    self.concentration_data[species] + min_val * 0.1
                )
                print(f"Applied log transformation to {species}")

# Example usage
pmf = CustomPMF()
pmf.load_data("data/Dataset-Baltimore_con.txt", "data/Dataset-Baltimore_unc.txt")
pmf.custom_missing_data_treatment(method='seasonal_median')
pmf.apply_log_transformation()
```

### Advanced Uncertainty Modeling

```python
def custom_uncertainty_calculation(concentration_data, method='hybrid'):
    """Calculate custom uncertainties using multiple methods."""

    uncertainties = pd.DataFrame(index=concentration_data.index,
                               columns=concentration_data.columns)

    for species in concentration_data.columns:
        conc = concentration_data[species]

        if method == 'hybrid':
            # Combine analytical and Poisson uncertainties
            analytical_unc = conc * 0.10  # 10% analytical uncertainty
            poisson_unc = np.sqrt(conc)   # Poisson counting uncertainty
            detection_limit = conc.quantile(0.05)  # 5th percentile as DL

            # Use the larger of analytical or Poisson uncertainty
            uncertainties[species] = np.maximum(analytical_unc, poisson_unc)

            # Apply minimum uncertainty at detection limit
            uncertainties[species] = np.maximum(uncertainties[species],
                                              detection_limit * 0.5)

        elif method == 'dynamic':
            # Concentration-dependent uncertainty
            low_conc_mask = conc < conc.quantile(0.25)
            med_conc_mask = (conc >= conc.quantile(0.25)) & (conc < conc.quantile(0.75))
            high_conc_mask = conc >= conc.quantile(0.75)

            uncertainties.loc[low_conc_mask, species] = conc[low_conc_mask] * 0.50  # 50% for low
            uncertainties.loc[med_conc_mask, species] = conc[med_conc_mask] * 0.15  # 15% for medium
            uncertainties.loc[high_conc_mask, species] = conc[high_conc_mask] * 0.10  # 10% for high

    return uncertainties

# Apply custom uncertainties
concentration_data = pmf.concentration_data
custom_uncertainties = custom_uncertainty_calculation(concentration_data, method='hybrid')
pmf.uncertainty_data = custom_uncertainties
```

## Advanced Factor Selection

### Multi-Criteria Factor Selection

```python
def advanced_factor_selection(pmf, factor_range=range(3, 10)):
    """Select optimal factors using multiple criteria."""

    results = {}
    metrics = {}

    for n_factors in factor_range:
        print(f"Testing {n_factors} factors...")

        result = pmf.run_pmf(n_factors=n_factors, n_runs=20)
        results[n_factors] = result

        # Calculate multiple metrics
        metrics[n_factors] = {
            'q_qexp': result.q_qexp,
            'q_robust': result.q_robust,
            'explained_variance': calculate_explained_variance(result),
            'factor_interpretability': assess_factor_interpretability(result),
            'stability_score': calculate_stability_score(result)
        }

    # Multi-criteria scoring
    scores = {}
    for n_factors in factor_range:
        m = metrics[n_factors]

        # Normalize metrics (lower is better for q_qexp, higher for others)
        q_score = 1 / (1 + abs(m['q_qexp'] - 1.0))  # Penalty for deviation from 1
        var_score = m['explained_variance'] / 100    # 0-1 scale
        interp_score = m['factor_interpretability']  # Already 0-1 scale
        stab_score = m['stability_score']            # Already 0-1 scale

        # Weighted combination
        scores[n_factors] = (
            0.3 * q_score +
            0.2 * var_score +
            0.3 * interp_score +
            0.2 * stab_score
        )

    # Select best scoring option
    optimal_factors = max(scores.keys(), key=lambda x: scores[x])

    print(f"\\nFactor Selection Results:")
    for n_factors in factor_range:
        m = metrics[n_factors]
        print(f"{n_factors} factors: Score={scores[n_factors]:.3f}, "
              f"Q/Qexp={m['q_qexp']:.2f}, ExplVar={m['explained_variance']:.1f}%")

    print(f"\\nOptimal choice: {optimal_factors} factors")
    return optimal_factors, results[optimal_factors]

def calculate_explained_variance(result):
    """Calculate percentage of variance explained by factors."""
    # This would need access to original data for proper calculation
    # Placeholder implementation
    return 85.0  # Example value

def assess_factor_interpretability(result):
    """Assess how interpretable the factors are."""
    profiles = result.factor_profiles

    # Count factors with clear dominant species
    interpretable_factors = 0
    for factor_idx in range(profiles.shape[1]):
        factor_profile = profiles.iloc[:, factor_idx]
        max_contribution = factor_profile.max()
        second_max = factor_profile.nlargest(2).iloc[1]

        # Factor is interpretable if one species dominates
        if max_contribution / second_max > 2.0:
            interpretable_factors += 1

    return interpretable_factors / profiles.shape[1]

def calculate_stability_score(result):
    """Calculate stability score based on multiple runs."""
    # This would compare results across multiple runs
    # Placeholder implementation
    return 0.85  # Example value

# Apply advanced factor selection
optimal_factors, optimal_result = advanced_factor_selection(pmf)
```

## Custom Visualization Workflows

### Interactive Dashboard Creation

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_interactive_dashboard(result, dataset_name="PMF Analysis"):
    """Create an interactive dashboard with multiple views."""

    # Get data
    profiles = result.factor_profiles
    contributions = result.factor_contributions

    # Create subplot structure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Factor Profiles Heatmap', 'Time Series Contributions',
                       'Factor Correlation', 'Top Species by Factor'),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "bar"}]]
    )

    # 1. Factor profiles heatmap
    fig.add_trace(
        go.Heatmap(
            z=profiles.values,
            x=[f"Factor {i+1}" for i in range(profiles.shape[1])],
            y=profiles.index,
            colorscale='Viridis',
            name="Profiles"
        ),
        row=1, col=1
    )

    # 2. Time series contributions
    for i, factor in enumerate(contributions.columns):
        fig.add_trace(
            go.Scatter(
                x=contributions.index,
                y=contributions[factor],
                mode='lines',
                name=f'Factor {i+1}',
                line=dict(width=2)
            ),
            row=1, col=2
        )

    # 3. Factor correlation heatmap
    correlation_matrix = contributions.corr()
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            name="Correlation"
        ),
        row=2, col=1
    )

    # 4. Top species by factor (show first factor as example)
    top_species = profiles.iloc[:, 0].nlargest(10)
    fig.add_trace(
        go.Bar(
            x=top_species.values,
            y=top_species.index,
            orientation='h',
            name="Factor 1 Top Species"
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"PMF Analysis Dashboard - {dataset_name}",
        height=800,
        showlegend=True
    )

    # Save interactive plot
    fig.write_html(f"output/{dataset_name}_interactive_dashboard.html")
    print(f"Interactive dashboard saved: {dataset_name}_interactive_dashboard.html")

    return fig

# Create interactive dashboard
dashboard = create_interactive_dashboard(optimal_result, "Baltimore")
```

### Custom Report Generation

```python
from jinja2 import Template
import base64
import io

def generate_custom_report(result, pmf, dataset_name, template_path=None):
    """Generate a custom HTML report with embedded plots."""

    if template_path is None:
        # Use built-in template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PMF Analysis Report - {{ dataset_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; }
                .section { margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px;
                         background-color: #e8f4fd; border-radius: 5px; }
                .plot { text-align: center; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PMF Analysis Report</h1>
                <h2>Dataset: {{ dataset_name }}</h2>
                <p>Generated: {{ generation_date }}</p>
            </div>

            <div class="section">
                <h3>Analysis Summary</h3>
                <div class="metric">
                    <strong>Optimal Factors:</strong> {{ optimal_factors }}
                </div>
                <div class="metric">
                    <strong>Q/Qexp:</strong> {{ q_qexp }}
                </div>
                <div class="metric">
                    <strong>Data Points:</strong> {{ n_samples }}
                </div>
                <div class="metric">
                    <strong>Species:</strong> {{ n_species }}
                </div>
            </div>

            <div class="section">
                <h3>Factor Profiles Summary</h3>
                <table>
                    <tr>
                        <th>Factor</th>
                        <th>Top Species</th>
                        <th>Contribution (%)</th>
                        <th>Interpretation</th>
                    </tr>
                    {% for factor_info in factor_summary %}
                    <tr>
                        <td>{{ factor_info.factor_name }}</td>
                        <td>{{ factor_info.top_species }}</td>
                        <td>{{ factor_info.contribution }}</td>
                        <td>{{ factor_info.interpretation }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <div class="section">
                <h3>Visualizations</h3>
                {% for plot in plots %}
                <div class="plot">
                    <h4>{{ plot.title }}</h4>
                    <img src="data:image/png;base64,{{ plot.data }}" style="max-width: 100%;">
                </div>
                {% endfor %}
            </div>

            <div class="section">
                <h3>Recommendations</h3>
                <ul>
                    {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
        template = Template(template_str)

    # Prepare data for template
    profiles = result.factor_profiles
    contributions = result.factor_contributions

    # Factor summaries
    factor_summary = []
    for i in range(profiles.shape[1]):
        factor_profile = profiles.iloc[:, i]
        top_species = factor_profile.nlargest(3)
        top_species_str = ", ".join([f"{species} ({val:.2f})"
                                   for species, val in top_species.items()])

        # Calculate factor contribution to total
        factor_contribution = contributions.iloc[:, i].sum() / contributions.sum().sum() * 100

        # Simple interpretation based on top species
        interpretation = interpret_factor(top_species.index.tolist())

        factor_summary.append({
            'factor_name': f'Factor {i+1}',
            'top_species': top_species_str,
            'contribution': f'{factor_contribution:.1f}%',
            'interpretation': interpretation
        })

    # Generate embedded plots
    plots = create_embedded_plots(result, dataset_name)

    # Generate recommendations
    recommendations = generate_recommendations(result, pmf)

    # Render template
    html_content = template.render(
        dataset_name=dataset_name,
        generation_date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        optimal_factors=profiles.shape[1],
        q_qexp=f'{result.q_qexp:.3f}',
        n_samples=pmf.concentration_data.shape[0],
        n_species=pmf.concentration_data.shape[1],
        factor_summary=factor_summary,
        plots=plots,
        recommendations=recommendations
    )

    # Save report
    output_file = f"output/{dataset_name}_custom_report.html"
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Custom report saved: {output_file}")
    return output_file

def interpret_factor(top_species):
    """Simple factor interpretation based on top species."""

    interpretation_rules = {
        ('EC', 'OC'): 'Traffic/Mobile Sources',
        ('SO4', 'NH4'): 'Secondary Sulfate',
        ('Al', 'Si', 'Ca'): 'Crustal/Dust',
        ('Zn', 'Pb', 'Cu'): 'Industrial Sources',
        ('K', 'OC'): 'Biomass Burning'
    }

    for rule_species, interpretation in interpretation_rules.items():
        if any(species in top_species for species in rule_species):
            return interpretation

    return 'Mixed/Unknown Source'

def create_embedded_plots(result, dataset_name):
    """Create plots as base64 encoded strings for embedding."""

    plots = []

    # Factor profiles heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    profiles = result.factor_profiles
    im = ax.imshow(profiles.T, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(profiles.index)))
    ax.set_xticklabels(profiles.index, rotation=45, ha='right')
    ax.set_yticks(range(len(profiles.columns)))
    ax.set_yticklabels([f'Factor {i+1}' for i in range(len(profiles.columns))])
    plt.colorbar(im)
    plt.title('Factor Profiles Heatmap')
    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plots.append({'title': 'Factor Profiles Heatmap', 'data': plot_data})
    plt.close()

    # Add more plots as needed...

    return plots

def generate_recommendations(result, pmf):
    """Generate analysis recommendations based on results."""

    recommendations = []

    # Check Q/Qexp
    if result.q_qexp > 1.5:
        recommendations.append("Consider increasing the number of factors or reviewing data quality (Q/Qexp > 1.5)")
    elif result.q_qexp < 0.8:
        recommendations.append("Model may be overfitting; consider reducing the number of factors (Q/Qexp < 0.8)")
    else:
        recommendations.append("Model fit is good (Q/Qexp ≈ 1.0)")

    # Check data quality
    missing_percentage = (pmf.concentration_data.isnull().sum() / len(pmf.concentration_data) * 100).max()
    if missing_percentage > 20:
        recommendations.append(f"High missing data percentage ({missing_percentage:.1f}%); consider data imputation methods")

    # Factor interpretation
    profiles = result.factor_profiles
    unclear_factors = 0
    for i in range(profiles.shape[1]):
        factor_profile = profiles.iloc[:, i]
        max_contribution = factor_profile.max()
        second_max = factor_profile.nlargest(2).iloc[1]
        if max_contribution / second_max < 1.5:
            unclear_factors += 1

    if unclear_factors > 0:
        recommendations.append(f"{unclear_factors} factor(s) may need clearer interpretation; consider factor rotation or different factor numbers")

    return recommendations

# Generate custom report
report_file = generate_custom_report(optimal_result, pmf, "Baltimore")
```

## Integration with External Tools

### Export to Other Software

```python
def export_for_epa_pmf(result, output_prefix="pmf_export"):
    """Export results in EPA PMF software format."""

    # Factor profiles
    profiles = result.factor_profiles
    profiles.to_csv(f"{output_prefix}_profiles.csv")

    # Factor contributions
    contributions = result.factor_contributions
    contributions.to_csv(f"{output_prefix}_contributions.csv")

    # Create EPA-style summary
    summary = {
        'Q_Qexp': result.q_qexp,
        'Q_robust': result.q_robust,
        'Number_of_factors': profiles.shape[1],
        'Number_of_species': profiles.shape[0],
        'Number_of_samples': contributions.shape[0]
    }

    pd.Series(summary).to_csv(f"{output_prefix}_summary.csv")
    print(f"Exported data for EPA PMF: {output_prefix}_*.csv")

def export_for_r_analysis(result, pmf, output_prefix="r_export"):
    """Export data for R-based analysis."""

    # Save all data in R-friendly format
    pmf.concentration_data.to_csv(f"{output_prefix}_concentrations.csv")
    pmf.uncertainty_data.to_csv(f"{output_prefix}_uncertainties.csv")
    result.factor_profiles.to_csv(f"{output_prefix}_profiles.csv")
    result.factor_contributions.to_csv(f"{output_prefix}_contributions.csv")

    # Create R script template
    r_script = f"""
# Load PMF results exported from Easy PMF
library(readr)
library(ggplot2)
library(corrplot)

# Load data
concentrations <- read_csv("{output_prefix}_concentrations.csv")
uncertainties <- read_csv("{output_prefix}_uncertainties.csv")
profiles <- read_csv("{output_prefix}_profiles.csv")
contributions <- read_csv("{output_prefix}_contributions.csv")

# Example visualizations
# Factor correlation plot
corrplot(cor(contributions[,-1]), method="circle")

# Time series plot
library(reshape2)
contrib_long <- melt(contributions, id.vars=1)
ggplot(contrib_long, aes(x=1:nrow(contributions), y=value, color=variable)) +
  geom_line() + facet_wrap(~variable, scales="free_y") +
  labs(x="Sample", y="Contribution", title="Factor Contributions Over Time")
"""

    with open(f"{output_prefix}_analysis.R", 'w') as f:
        f.write(r_script)

    print(f"Exported data for R analysis: {output_prefix}_*.csv and {output_prefix}_analysis.R")

# Export results
export_for_epa_pmf(optimal_result)
export_for_r_analysis(optimal_result, pmf)
```

## Performance Optimization

### Parallel Processing

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

def parallel_pmf_analysis(pmf_data, factor_range, n_runs=20, n_processes=None):
    """Run PMF analysis in parallel for different factor numbers."""

    if n_processes is None:
        n_processes = min(mp.cpu_count() - 1, len(factor_range))

    def run_single_analysis(n_factors):
        """Run PMF for a single factor number."""
        pmf_copy = pmf_data.copy()  # Ensure thread safety
        result = pmf_copy.run_pmf(n_factors=n_factors, n_runs=n_runs)
        return n_factors, result

    print(f"Running parallel analysis with {n_processes} processes...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(run_single_analysis, n_factors)
                  for n_factors in factor_range]

        results = {}
        for future in futures:
            n_factors, result = future.result()
            results[n_factors] = result
            print(f"Completed {n_factors} factors: Q/Qexp = {result.q_qexp:.3f}")

    end_time = time.time()
    print(f"Parallel analysis completed in {end_time - start_time:.1f} seconds")

    return results

# Run parallel analysis
parallel_results = parallel_pmf_analysis(pmf, range(3, 10), n_runs=20)
```

This comprehensive guide provides advanced techniques for customizing PMF analysis workflows. These methods allow for sophisticated analysis approaches tailored to specific research needs and datasets.

For basic usage, see the [Baltimore example](baltimore.md) or [batch processing guide](batch-processing.md).
