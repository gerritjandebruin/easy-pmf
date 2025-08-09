# Interpreting Results

This guide explains how to interpret and validate PMF analysis results to gain meaningful insights about pollution sources.

## Understanding PMF Outputs

### Factor Contributions (G Matrix)

The factor contributions tell you **when and how much** each source contributed to the measured concentrations.

```python
# Access factor contributions
contributions = pmf.contributions_
print(contributions.head())

#              Factor_1  Factor_2  Factor_3  Factor_4  Factor_5
# 2023-01-01      12.3      8.7      2.1      15.6      4.2
# 2023-01-02       9.8      6.2      1.9      18.2      3.8
# 2023-01-03      14.1      9.3      2.5      12.4      5.1
```

**Key interpretations**:
- Each row represents a time point (sample)
- Each column represents a pollution source/factor
- Values are mass contributions (e.g., μg/m³)
- Higher values = stronger source influence at that time

### Factor Profiles (F Matrix)

The factor profiles show the **chemical fingerprint** of each source.

```python
# Access factor profiles
profiles = pmf.profiles_
print(profiles.head())

#           PM2.5   SO4   NO3    EC    OC    Na    Cl
# Factor_1   0.15  0.45  0.02  0.08  0.12  0.01  0.01  # Coal combustion
# Factor_2   0.20  0.05  0.25  0.18  0.22  0.02  0.02  # Traffic
# Factor_3   0.05  0.12  0.08  0.02  0.05  0.35  0.28  # Sea salt
```

**Key interpretations**:
- Each row represents a source/factor
- Each column represents a chemical species
- Values show relative abundance of each species in the source
- High values indicate "marker species" for that source

## Source Identification

### Chemical Signatures

Use chemical knowledge to identify sources based on their profiles:

```python
import pandas as pd
import matplotlib.pyplot as plt

def identify_sources(profiles):
    """Help identify sources based on chemical signatures."""

    # Define marker species for common sources
    markers = {
        'Traffic': ['EC', 'OC', 'NO3', 'Cu', 'Zn'],
        'Coal': ['SO4', 'As', 'Se'],
        'Sea Salt': ['Na', 'Cl', 'Mg'],
        'Soil': ['Al', 'Si', 'Ca', 'Fe', 'Ti'],
        'Secondary Sulfate': ['SO4', 'NH4'],
        'Oil Combustion': ['V', 'Ni'],
        'Biomass Burning': ['K', 'OC']
    }

    # Calculate enrichment for each source type
    source_scores = {}

    for factor_name in profiles.index:
        factor_profile = profiles.loc[factor_name]

        scores = {}
        for source_type, marker_species in markers.items():
            # Find markers present in data
            available_markers = [s for s in marker_species if s in factor_profile.index]

            if available_markers:
                # Calculate average relative abundance of markers
                marker_values = factor_profile[available_markers]
                scores[source_type] = marker_values.mean()
            else:
                scores[source_type] = 0

        source_scores[factor_name] = scores

    # Convert to DataFrame for easy viewing
    identification = pd.DataFrame(source_scores).T

    # Find most likely source for each factor
    likely_sources = identification.idxmax(axis=1)

    print("Factor identification based on chemical markers:")
    print("=" * 50)
    for factor, source in likely_sources.items():
        score = identification.loc[factor, source]
        print(f"{factor}: {source} (score: {score:.3f})")

    return identification, likely_sources

# Identify sources
source_id, likely_sources = identify_sources(pmf.profiles_)
```

### Visualization for Source ID

```python
def plot_factor_profiles_with_markers(profiles, factor_names=None):
    """Plot factor profiles highlighting marker species."""

    n_factors = len(profiles)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Define colors for different source types
    marker_colors = {
        'Traffic': ['EC', 'OC', 'NO3', 'Cu', 'Zn'],
        'Coal': ['SO4', 'As', 'Se'],
        'Sea Salt': ['Na', 'Cl', 'Mg'],
        'Soil': ['Al', 'Si', 'Ca', 'Fe', 'Ti']
    }

    for i, (factor_idx, factor_data) in enumerate(profiles.iterrows()):
        if i >= len(axes):
            break

        ax = axes[i]

        # Base bar plot
        bars = ax.bar(range(len(factor_data)), factor_data, alpha=0.7)

        # Highlight marker species
        for source_type, markers in marker_colors.items():
            for j, species in enumerate(factor_data.index):
                if species in markers:
                    bars[j].set_color('red')
                    bars[j].set_alpha(0.9)

        ax.set_title(f'Factor {i+1}' + (f' ({factor_names[i]})' if factor_names else ''))
        ax.set_xlabel('Chemical Species')
        ax.set_ylabel('Relative Abundance')
        ax.set_xticks(range(len(factor_data)))
        ax.set_xticklabels(factor_data.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for i in range(n_factors, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Plot with marker highlighting
plot_factor_profiles_with_markers(pmf.profiles_)
```

## Temporal Pattern Analysis

### Seasonal Patterns

```python
def analyze_seasonal_patterns(contributions):
    """Analyze seasonal patterns in factor contributions."""

    # Add time components
    df = contributions.copy()
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['hour'] = df.index.hour if hasattr(df.index, 'hour') else None

    # Monthly patterns
    monthly_avg = df.groupby('month')[contributions.columns].mean()

    # Weekday patterns
    weekday_avg = df.groupby('weekday')[contributions.columns].mean()

    # Plot seasonal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Monthly patterns
    monthly_avg.T.plot(kind='bar', ax=axes[0, 0], legend=False)
    axes[0, 0].set_title('Monthly Average Contributions')
    axes[0, 0].set_xlabel('Factor')
    axes[0, 0].set_ylabel('Average Contribution')
    axes[0, 0].legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Weekday patterns
    weekday_avg.T.plot(kind='bar', ax=axes[0, 1], legend=False)
    axes[0, 1].set_title('Weekday Average Contributions')
    axes[0, 1].set_xlabel('Factor')
    axes[0, 1].set_ylabel('Average Contribution')
    axes[0, 1].legend(title='Weekday', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Time series of dominant factors
    dominant_factor = contributions.idxmax(axis=1)
    factor_counts = dominant_factor.value_counts()

    axes[1, 0].pie(factor_counts.values, labels=factor_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Dominant Factor Distribution')

    # Correlation between factors
    factor_corr = contributions.corr()
    im = axes[1, 1].imshow(factor_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(factor_corr)))
    axes[1, 1].set_yticks(range(len(factor_corr)))
    axes[1, 1].set_xticklabels(factor_corr.columns)
    axes[1, 1].set_yticklabels(factor_corr.index)
    axes[1, 1].set_title('Factor Correlation Matrix')

    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    return monthly_avg, weekday_avg

# Analyze patterns
monthly_patterns, weekday_patterns = analyze_seasonal_patterns(pmf.contributions_)
```

### Source Strength Variability

```python
def analyze_source_variability(contributions):
    """Analyze variability and statistics of source contributions."""

    stats = pd.DataFrame({
        'Mean': contributions.mean(),
        'Std': contributions.std(),
        'Min': contributions.min(),
        'Max': contributions.max(),
        'CV': contributions.std() / contributions.mean(),  # Coefficient of variation
        'Contribution_%': contributions.mean() / contributions.sum(axis=1).mean() * 100
    })

    print("Source Contribution Statistics:")
    print("=" * 40)
    print(stats.round(3))

    # Plot contribution statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Mean contributions
    stats['Mean'].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Mean Factor Contributions')
    axes[0, 0].set_ylabel('Average Contribution')

    # Coefficient of variation
    stats['CV'].plot(kind='bar', ax=axes[0, 1], color='orange')
    axes[0, 1].set_title('Coefficient of Variation')
    axes[0, 1].set_ylabel('CV (std/mean)')

    # Percentage contribution
    stats['Contribution_%'].plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%')
    axes[1, 0].set_title('Relative Source Contributions')

    # Box plot of all contributions
    contributions.boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('Contribution Distributions')
    axes[1, 1].set_ylabel('Contribution')

    plt.tight_layout()
    plt.show()

    return stats

# Analyze source variability
source_stats = analyze_source_variability(pmf.contributions_)
```

## Model Quality Assessment

### Residual Analysis

```python
def analyze_residuals(concentrations, uncertainties, pmf):
    """Comprehensive residual analysis."""

    # Calculate residuals
    reconstructed = pmf.contributions_.values @ pmf.profiles_.values
    residuals = concentrations.values - reconstructed
    scaled_residuals = residuals / uncertainties.values

    print("Residual Analysis:")
    print("=" * 30)
    print(f"Q-value: {pmf.score(concentrations, uncertainties):.2f}")

    # Theoretical Q-value
    n_samples, n_species = concentrations.shape
    n_factors = pmf.n_components
    q_theoretical = (n_samples * n_species) - (n_factors * (n_samples + n_species))
    q_ratio = pmf.score(concentrations, uncertainties) / q_theoretical

    print(f"Theoretical Q: {q_theoretical:.0f}")
    print(f"Q/Q_theoretical: {q_ratio:.2f}")

    # Scaled residual statistics
    print(f"\nScaled Residual Statistics:")
    print(f"Mean: {np.mean(scaled_residuals):.3f}")
    print(f"Std: {np.std(scaled_residuals):.3f}")
    print(f"% with |residual| > 3: {(np.abs(scaled_residuals) > 3).mean() * 100:.1f}%")

    # Plot residual analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Scaled residuals histogram
    axes[0, 0].hist(scaled_residuals.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--')
    x = np.linspace(-4, 4, 100)
    axes[0, 0].plot(x, norm.pdf(x, 0, 1), 'r-', label='Normal(0,1)')
    axes[0, 0].set_xlabel('Scaled Residuals')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Scaled Residuals Distribution')
    axes[0, 0].legend()

    # Q-Q plot
    from scipy import stats
    stats.probplot(scaled_residuals.flatten(), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot vs Normal Distribution')

    # Residuals vs reconstructed
    axes[0, 2].scatter(reconstructed.flatten(), residuals.flatten(), alpha=0.5)
    axes[0, 2].axhline(0, color='red', linestyle='--')
    axes[0, 2].set_xlabel('Reconstructed Concentration')
    axes[0, 2].set_ylabel('Residual')
    axes[0, 2].set_title('Residuals vs Reconstructed')

    # Time series of residuals (for first few species)
    for i, species in enumerate(concentrations.columns[:3]):
        axes[1, i].plot(concentrations.index, scaled_residuals[:, i])
        axes[1, i].axhline(0, color='red', linestyle='--')
        axes[1, i].axhline(3, color='orange', linestyle='--', alpha=0.7)
        axes[1, i].axhline(-3, color='orange', linestyle='--', alpha=0.7)
        axes[1, i].set_xlabel('Date')
        axes[1, i].set_ylabel('Scaled Residual')
        axes[1, i].set_title(f'Scaled Residuals: {species}')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'q_value': pmf.score(concentrations, uncertainties),
        'q_theoretical': q_theoretical,
        'q_ratio': q_ratio,
        'residual_stats': {
            'mean': np.mean(scaled_residuals),
            'std': np.std(scaled_residuals),
            'outlier_pct': (np.abs(scaled_residuals) > 3).mean() * 100
        }
    }

from scipy.stats import norm
import numpy as np

# Run residual analysis
residual_results = analyze_residuals(concentrations, uncertainties, pmf)
```

### Species Reconstruction

```python
def analyze_species_reconstruction(concentrations, pmf):
    """Analyze how well each species is reconstructed."""

    reconstructed = pd.DataFrame(
        pmf.contributions_.values @ pmf.profiles_.values,
        index=concentrations.index,
        columns=concentrations.columns
    )

    # Calculate R² for each species
    r_squared = {}
    for species in concentrations.columns:
        obs = concentrations[species]
        pred = reconstructed[species]

        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r_squared[species] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    r_squared_df = pd.Series(r_squared, name='R²').sort_values(ascending=False)

    print("Species Reconstruction Quality (R²):")
    print("=" * 35)
    for species, r2 in r_squared_df.items():
        print(f"{species:>10}: {r2:.3f}")

    # Plot reconstruction quality
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # R² bar plot
    r_squared_df.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Reconstruction Quality by Species')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Scatter plot: observed vs predicted (all species)
    axes[0, 1].scatter(concentrations.values.flatten(),
                      reconstructed.values.flatten(), alpha=0.5)

    # Add 1:1 line
    min_val = min(concentrations.min().min(), reconstructed.min().min())
    max_val = max(concentrations.max().max(), reconstructed.max().max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

    axes[0, 1].set_xlabel('Observed Concentration')
    axes[0, 1].set_ylabel('Reconstructed Concentration')
    axes[0, 1].set_title('Observed vs Reconstructed (All Species)')

    # Time series comparison for best and worst species
    best_species = r_squared_df.index[0]
    worst_species = r_squared_df.index[-1]

    # Best species
    axes[1, 0].plot(concentrations.index, concentrations[best_species],
                   label='Observed', alpha=0.8)
    axes[1, 0].plot(reconstructed.index, reconstructed[best_species],
                   label='Reconstructed', alpha=0.8)
    axes[1, 0].set_title(f'Best Reconstruction: {best_species} (R²={r_squared_df[best_species]:.3f})')
    axes[1, 0].set_ylabel('Concentration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Worst species
    axes[1, 1].plot(concentrations.index, concentrations[worst_species],
                   label='Observed', alpha=0.8)
    axes[1, 1].plot(reconstructed.index, reconstructed[worst_species],
                   label='Reconstructed', alpha=0.8)
    axes[1, 1].set_title(f'Worst Reconstruction: {worst_species} (R²={r_squared_df[worst_species]:.3f})')
    axes[1, 1].set_ylabel('Concentration')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return r_squared_df, reconstructed

# Analyze species reconstruction
r_squared_results, reconstructed_data = analyze_species_reconstruction(concentrations, pmf)
```

## Validation Against External Data

### Meteorological Validation

```python
def validate_with_meteorology(contributions, met_data=None):
    """Validate PMF results against meteorological data."""

    if met_data is None:
        print("No meteorological data provided for validation")
        return

    # Example validation approaches:
    # 1. Wind direction analysis for local vs regional sources
    # 2. Temperature correlation for secondary aerosol formation
    # 3. Precipitation effects on source contributions

    # Placeholder for meteorological validation
    print("Meteorological validation would include:")
    print("- Wind rose analysis for source directionality")
    print("- Temperature correlation with secondary sources")
    print("- Precipitation effects on contributions")
    print("- Boundary layer height impacts")

# Example call (requires meteorological data)
# validate_with_meteorology(pmf.contributions_, met_data)
```

### Emission Inventory Comparison

```python
def compare_with_emissions(contributions, emission_data=None):
    """Compare PMF results with emission inventory data."""

    if emission_data is None:
        print("No emission inventory data provided for comparison")
        return

    # Example comparison approaches:
    # 1. Seasonal patterns comparison
    # 2. Spatial correlation analysis
    # 3. Source contribution percentages

    print("Emission inventory comparison would include:")
    print("- Seasonal pattern consistency")
    print("- Source contribution percentages")
    print("- Spatial distribution validation")
    print("- Trend analysis over time")

# Example call (requires emission data)
# compare_with_emissions(pmf.contributions_, emission_data)
```

## Reporting and Communication

### Summary Report Generation

```python
def generate_pmf_report(pmf, concentrations, uncertainties, site_name="Unknown"):
    """Generate a comprehensive PMF analysis report."""

    report = []
    report.append(f"PMF Analysis Report: {site_name}")
    report.append("=" * (len(f"PMF Analysis Report: {site_name}")))

    # Model parameters
    report.append(f"\nModel Parameters:")
    report.append(f"  Number of factors: {pmf.n_components}")
    report.append(f"  Convergence: {pmf.converged_}")
    report.append(f"  Iterations: {pmf.n_iter_}")
    report.append(f"  Q-value: {pmf.score(concentrations, uncertainties):.2f}")

    # Data summary
    report.append(f"\nData Summary:")
    report.append(f"  Samples: {len(concentrations)}")
    report.append(f"  Species: {len(concentrations.columns)}")
    report.append(f"  Date range: {concentrations.index.min()} to {concentrations.index.max()}")

    # Factor contributions summary
    contrib_summary = pmf.contributions_.describe()
    report.append(f"\nFactor Contribution Summary:")
    report.append(f"  Mean contributions: {pmf.contributions_.mean().round(2).to_dict()}")
    report.append(f"  Contribution percentages: {(pmf.contributions_.mean() / pmf.contributions_.sum(axis=1).mean() * 100).round(1).to_dict()}")

    # Key findings
    report.append(f"\nKey Findings:")
    dominant_factor = pmf.contributions_.mean().idxmax()
    dominant_pct = (pmf.contributions_.mean() / pmf.contributions_.sum(axis=1).mean() * 100).max()
    report.append(f"  Dominant source: {dominant_factor} ({dominant_pct:.1f}%)")

    # Print report
    full_report = "\n".join(report)
    print(full_report)

    return full_report

# Generate report
report = generate_pmf_report(pmf, concentrations, uncertainties, "Example Site")
```

## Best Practices for Interpretation

### 1. Use Multiple Lines of Evidence
- Chemical signatures from profiles
- Temporal patterns from contributions
- External validation data
- Literature comparisons

### 2. Consider Uncertainties
- Bootstrap confidence intervals
- Model stability across runs
- Measurement uncertainties

### 3. Apply Domain Knowledge
- Known sources in the study area
- Seasonal patterns expectations
- Chemical process understanding

### 4. Validate Results
- Residual analysis
- External data comparison
- Sensitivity testing

### 5. Communicate Clearly
- Use descriptive source names
- Provide uncertainty estimates
- Include validation evidence
- Explain limitations

## Common Interpretation Pitfalls

### ❌ Avoid These Mistakes

1. **Over-interpretation**: Don't assign source names without chemical evidence
2. **Ignoring uncertainty**: Always consider bootstrap confidence intervals
3. **Single metric focus**: Don't rely only on Q-value for model quality
4. **Missing validation**: Always validate against external information
5. **Factor splitting**: Be careful about too many factors creating artificial splits

### ✅ Best Practices

1. **Use chemical knowledge**: Base interpretations on sound chemical principles
2. **Multiple validation**: Use several validation approaches
3. **Report uncertainty**: Include confidence intervals and limitations
4. **Document assumptions**: Clearly state interpretation assumptions
5. **Peer review**: Have domain experts review interpretations

## Next Steps

- Learn about [Visualization](visualization.md) techniques for presenting results
- Explore [Advanced Examples](../examples/baltimore.md) with real data
- Review [Contributing Guidelines](../contributing/guidelines.md) to improve the package
