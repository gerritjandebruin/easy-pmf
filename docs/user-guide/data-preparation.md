# Data Preparation

Proper data preparation is crucial for successful PMF analysis. This guide covers all aspects of preparing your environmental data for PMF.

## Data Requirements Overview

### Essential Data Components

1. **Concentration Matrix (X)**: Chemical species concentrations
2. **Uncertainty Matrix (U)**: Measurement uncertainties (optional but recommended)
3. **Metadata**: Sample dates, locations, additional information

### Data Format Specifications

```python
import pandas as pd
import numpy as np

# Ideal data structure
concentrations = pd.DataFrame({
    'Species_1': [12.5, 8.3, 15.7, ...],    # µg/m³ or appropriate units
    'Species_2': [0.8, 1.2, 0.5, ...],
    'Species_3': [25.1, 18.9, 32.4, ...],
    # ... more species
}, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', ...]))
```

## Data Collection Guidelines

### Sampling Strategy

**Temporal Coverage**:
- Minimum 50-100 samples for stable PMF
- Preferably 200+ samples for robust results
- Cover seasonal cycles if possible
- Include both weekdays and weekends

**Species Selection**:
- Include 10-30 chemical species
- Mix of primary and secondary pollutants
- Include source-specific tracers
- Avoid highly correlated redundant species

### Quality Criteria

| Criterion | Recommendation | Impact if Violated |
|-----------|----------------|-------------------|
| Detection Rate | >50% above detection limit | Poor factor profiles |
| Signal-to-Noise | S/N > 2 | Noisy results |
| Dynamic Range | >10:1 (max/min) | Poor resolution |
| Missing Data | <25% missing | Biased results |

## Data Loading and Initial Processing

### Loading Different File Formats

```python
import pandas as pd

# CSV files
data = pd.read_csv('concentrations.csv', index_col=0, parse_dates=True)

# Tab-separated files
data = pd.read_csv('data.txt', sep='\t', index_col=0, parse_dates=True)

# Excel files
data = pd.read_excel('data.xlsx', sheet_name='Concentrations', index_col=0)

# Multiple sheets
conc = pd.read_excel('data.xlsx', sheet_name='Concentrations', index_col=0)
unc = pd.read_excel('data.xlsx', sheet_name='Uncertainties', index_col=0)
```

### Initial Data Inspection

```python
def inspect_data(df, name="Data"):
    """Comprehensive data inspection."""
    print(f"\n=== {name} Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Species: {list(df.columns)}")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:\n{missing[missing > 0]}")

    # Negative values
    negative = (df < 0).sum()
    if negative.any():
        print(f"\nNegative values:\n{negative[negative > 0]}")

    # Zero values
    zeros = (df == 0).sum()
    print(f"\nZero values:\n{zeros}")

    # Basic statistics
    print(f"\nBasic statistics:\n{df.describe()}")

# Use the inspector
inspect_data(concentrations, "Concentrations")
inspect_data(uncertainties, "Uncertainties")
```

## Data Quality Control

### Outlier Detection and Treatment

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_outliers(df, method='iqr', threshold=3):
    """Detect outliers using different methods."""
    outliers = pd.DataFrame(index=df.index, columns=df.columns, dtype=bool)

    if method == 'iqr':
        # Interquartile Range method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > threshold

    elif method == 'modified_zscore':
        # Modified Z-score using median
        median = df.median()
        mad = np.median(np.abs(df - median))
        modified_z_scores = 0.6745 * (df - median) / mad
        outliers = np.abs(modified_z_scores) > threshold

    return outliers

# Detect outliers
outliers = detect_outliers(concentrations, method='iqr')
print(f"Outliers detected: {outliers.sum().sum()}")

# Visualize outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
species_sample = concentrations.columns[:4]

for i, species in enumerate(species_sample):
    ax = axes[i//2, i%2]

    # Box plot
    concentrations[species].plot(kind='box', ax=ax)
    ax.set_title(f'{species} - Outlier Detection')
    ax.set_ylabel('Concentration')

plt.tight_layout()
plt.show()
```

### Missing Value Treatment

```python
def handle_missing_values(df, method='interpolate', limit=None):
    """Handle missing values with different strategies."""

    if method == 'drop':
        # Remove samples with any missing values
        return df.dropna()

    elif method == 'drop_species':
        # Remove species with too many missing values
        missing_pct = df.isnull().mean()
        keep_species = missing_pct < 0.25  # Keep species with <25% missing
        return df.loc[:, keep_species]

    elif method == 'interpolate':
        # Linear interpolation
        return df.interpolate(method='linear', limit=limit)

    elif method == 'forward_fill':
        # Forward fill
        return df.fillna(method='ffill', limit=limit)

    elif method == 'median':
        # Replace with median
        return df.fillna(df.median())

    elif method == 'detection_limit':
        # Replace with detection limit (typically DL/2)
        detection_limits = df.quantile(0.1)  # Approximation
        return df.fillna(detection_limits / 2)

# Handle missing values
concentrations_clean = handle_missing_values(concentrations, method='interpolate', limit=3)
```

### Below Detection Limit (BDL) Treatment

```python
def handle_bdl_values(df, detection_limits=None, method='half_dl'):
    """Handle below detection limit values."""

    if detection_limits is None:
        # Estimate detection limits as 10th percentile
        detection_limits = df.quantile(0.1)

    if method == 'half_dl':
        # Replace zeros/BDL with DL/2
        df_clean = df.copy()
        for col in df.columns:
            bdl_mask = df[col] <= detection_limits[col]
            df_clean.loc[bdl_mask, col] = detection_limits[col] / 2

    elif method == 'dl_sqrt2':
        # Replace with DL/√2
        df_clean = df.copy()
        for col in df.columns:
            bdl_mask = df[col] <= detection_limits[col]
            df_clean.loc[bdl_mask, col] = detection_limits[col] / np.sqrt(2)

    elif method == 'small_positive':
        # Replace with small positive value
        df_clean = df.copy()
        df_clean[df_clean <= 0] = 1e-6

    return df_clean

# Handle BDL values
concentrations_bdl = handle_bdl_values(concentrations_clean)
```

## Species Selection and Filtering

### Detection Rate Filtering

```python
def filter_by_detection_rate(df, min_detection_rate=0.5):
    """Keep species with sufficient detection rate."""
    detection_rates = (df > 0).mean()
    valid_species = detection_rates >= min_detection_rate

    print(f"Species detection rates:")
    for species, rate in detection_rates.items():
        status = "✓" if rate >= min_detection_rate else "✗"
        print(f"  {status} {species}: {rate:.2%}")

    return df.loc[:, valid_species]

# Filter by detection rate
concentrations_filtered = filter_by_detection_rate(concentrations_bdl, min_detection_rate=0.5)
```

### Signal-to-Noise Ratio

```python
def calculate_snr(concentrations, uncertainties):
    """Calculate signal-to-noise ratio for each species."""
    snr = concentrations.mean() / uncertainties.mean()
    return snr

def filter_by_snr(concentrations, uncertainties, min_snr=2.0):
    """Filter species by signal-to-noise ratio."""
    snr = calculate_snr(concentrations, uncertainties)
    valid_species = snr >= min_snr

    print(f"Signal-to-noise ratios:")
    for species, ratio in snr.items():
        status = "✓" if ratio >= min_snr else "✗"
        print(f"  {status} {species}: {ratio:.1f}")

    return concentrations.loc[:, valid_species], uncertainties.loc[:, valid_species]

# Filter by SNR
if uncertainties is not None:
    conc_snr, unc_snr = filter_by_snr(concentrations_filtered, uncertainties, min_snr=2.0)
else:
    conc_snr = concentrations_filtered
    unc_snr = None
```

### Correlation Analysis

```python
import seaborn as sns

def analyze_correlations(df, threshold=0.95):
    """Identify highly correlated species pairs."""
    corr_matrix = df.corr().abs()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    if high_corr_pairs:
        print(f"Highly correlated pairs (r > {threshold}):")
        for sp1, sp2, corr in high_corr_pairs:
            print(f"  {sp1} - {sp2}: r = {corr:.3f}")

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                center=0, cmap='RdBu_r', vmax=1, vmin=-1)
    plt.title('Species Correlation Matrix')
    plt.tight_layout()
    plt.show()

    return high_corr_pairs

# Analyze correlations
high_corr = analyze_correlations(conc_snr, threshold=0.95)
```

## Uncertainty Estimation

### When Uncertainties Are Not Available

```python
def estimate_uncertainties(concentrations, method='percentage'):
    """Estimate uncertainties when not directly available."""

    if method == 'percentage':
        # Use percentage of concentration (typically 10-30%)
        uncertainties = concentrations * 0.15  # 15% uncertainty

    elif method == 'sqrt':
        # Square root of concentration (Poisson-like)
        uncertainties = np.sqrt(concentrations.clip(lower=1))

    elif method == 'constant_cv':
        # Constant coefficient of variation
        cv = 0.2  # 20% CV
        uncertainties = concentrations * cv

    elif method == 'detection_based':
        # Based on detection limits
        detection_limits = concentrations.quantile(0.1)
        uncertainties = pd.DataFrame(index=concentrations.index,
                                   columns=concentrations.columns)

        for col in concentrations.columns:
            dl = detection_limits[col]
            # Higher uncertainty for low concentrations
            uncertainties[col] = np.where(
                concentrations[col] <= 3 * dl,
                concentrations[col] * 0.5,  # 50% for low concentrations
                concentrations[col] * 0.15   # 15% for higher concentrations
            )

    # Ensure minimum uncertainty
    uncertainties = uncertainties.clip(lower=0.01)

    return uncertainties

# Estimate uncertainties if not available
if unc_snr is None:
    unc_estimated = estimate_uncertainties(conc_snr, method='detection_based')
    print("Uncertainties estimated using detection-based method")
else:
    unc_estimated = unc_snr
```

### Uncertainty Validation

```python
def validate_uncertainties(concentrations, uncertainties):
    """Validate uncertainty estimates."""

    # Check for zero or negative uncertainties
    invalid_unc = (uncertainties <= 0).sum()
    if invalid_unc.any():
        print(f"Warning: Zero/negative uncertainties found:\n{invalid_unc[invalid_unc > 0]}")

    # Check uncertainty-to-concentration ratios
    rel_unc = uncertainties / concentrations
    rel_unc_median = rel_unc.median()

    print("Relative uncertainty statistics:")
    print(f"Median relative uncertainty by species:\n{rel_unc_median}")

    # Plot uncertainty relationships
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Uncertainty vs concentration
    for col in concentrations.columns[:5]:  # Show first 5 species
        axes[0].scatter(concentrations[col], uncertainties[col],
                       alpha=0.6, label=col)
    axes[0].set_xlabel('Concentration')
    axes[0].set_ylabel('Uncertainty')
    axes[0].set_title('Uncertainty vs Concentration')
    axes[0].legend()

    # Relative uncertainty distribution
    rel_unc.boxplot(ax=axes[1])
    axes[1].set_title('Relative Uncertainty Distribution')
    axes[1].set_ylabel('Uncertainty / Concentration')

    plt.tight_layout()
    plt.show()

# Validate uncertainties
validate_uncertainties(conc_snr, unc_estimated)
```

## Final Data Preparation

### Data Summary and Export

```python
def prepare_final_dataset(concentrations, uncertainties,
                         output_dir='prepared_data'):
    """Prepare final cleaned dataset for PMF analysis."""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Final validation
    print("=== Final Dataset Summary ===")
    print(f"Concentration matrix shape: {concentrations.shape}")
    print(f"Uncertainty matrix shape: {uncertainties.shape}")
    print(f"Date range: {concentrations.index.min()} to {concentrations.index.max()}")
    print(f"Species: {list(concentrations.columns)}")

    # Check for any remaining issues
    issues = []
    if (concentrations < 0).any().any():
        issues.append("Negative concentrations")
    if concentrations.isnull().any().any():
        issues.append("Missing concentrations")
    if (uncertainties <= 0).any().any():
        issues.append("Invalid uncertainties")
    if uncertainties.isnull().any().any():
        issues.append("Missing uncertainties")

    if issues:
        print(f"\nRemaining issues: {', '.join(issues)}")
    else:
        print("\n✓ Dataset ready for PMF analysis!")

    # Save cleaned data
    concentrations.to_csv(f'{output_dir}/concentrations_clean.csv')
    uncertainties.to_csv(f'{output_dir}/uncertainties_clean.csv')

    # Save metadata
    metadata = {
        'n_samples': len(concentrations),
        'n_species': len(concentrations.columns),
        'date_range': f"{concentrations.index.min()} to {concentrations.index.max()}",
        'species': list(concentrations.columns),
        'preparation_date': pd.Timestamp.now().isoformat()
    }

    import json
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nFiles saved to {output_dir}/")

    return concentrations, uncertainties

# Prepare final dataset
final_conc, final_unc = prepare_final_dataset(conc_snr, unc_estimated)
```

### Quality Control Checklist

Before proceeding to PMF analysis, verify:

- [ ] **Non-negative concentrations**: All values ≥ 0
- [ ] **No missing values**: All NaN values handled
- [ ] **Positive uncertainties**: All uncertainty values > 0
- [ ] **Consistent dimensions**: Concentration and uncertainty matrices same shape
- [ ] **Sufficient samples**: At least 50-100 samples
- [ ] **Good species selection**: 10-30 species with good detection rates
- [ ] **Reasonable uncertainties**: Relative uncertainties typically 10-50%
- [ ] **Date/time index**: Proper temporal indexing
- [ ] **Units consistency**: All concentrations in same units
- [ ] **Outliers addressed**: Extreme values investigated and handled

## Common Data Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| High missing data | >25% missing for some species | Drop species or use imputation |
| Negative values | Negative concentrations | Check data quality, set to zero or small positive |
| Poor detection | >50% below detection limit | Consider removing species |
| High correlation | r > 0.95 between species | Remove redundant species |
| Large uncertainties | Rel. uncertainty > 100% | Check measurement methods |
| Temporal gaps | Irregular sampling | Interpolate or analyze gaps separately |

## Next Steps

With properly prepared data, you can proceed to:

- [Running PMF Analysis](running-analysis.md)
- [Model Selection and Validation](../examples/baltimore.md)
- [Results Interpretation](interpreting-results.md)
