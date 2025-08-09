# Baltimore Dataset Analysis Example

This example demonstrates a complete PMF analysis workflow using the Baltimore dataset, one of the sample datasets included with Easy PMF.

## Overview

The Baltimore dataset contains atmospheric particulate matter measurements with concentrations and uncertainties for various chemical species. This example will walk you through:

1. Loading and preparing the Baltimore dataset
2. Running PMF analysis with different factor numbers
3. Interpreting the results
4. Visualizing factor profiles and contributions

## Dataset Information

The Baltimore dataset includes:
- **File**: `data/Dataset-Baltimore_con.txt` (concentrations)
- **Uncertainty File**: `data/Dataset-Baltimore_unc.txt` (uncertainties)
- **Species**: Multiple chemical species including metals, ions, and organic compounds
- **Samples**: Time series of atmospheric measurements

## Step-by-Step Analysis

### 1. Loading the Data

```python
from easy_pmf import PMF
import pandas as pd

# Initialize PMF with Baltimore dataset
pmf = PMF()

# Load concentration and uncertainty data
pmf.load_data(
    concentration_file="data/Dataset-Baltimore_con.txt",
    uncertainty_file="data/Dataset-Baltimore_unc.txt"
)

# Display basic information about the dataset
print(f"Dataset shape: {pmf.concentration_data.shape}")
print(f"Species: {list(pmf.concentration_data.columns)}")
print(f"Sample period: {len(pmf.concentration_data)} samples")
```

### 2. Data Preparation

```python
# Check for missing values and data quality
pmf.check_data_quality()

# Prepare data for PMF analysis
pmf.prepare_data(
    remove_missing_threshold=0.5,  # Remove species with >50% missing data
    replace_missing_method="median",  # Replace missing values with median
    apply_uncertainty_scaling=True
)
```

### 3. Running PMF Analysis

```python
# Test different numbers of factors
factor_range = range(3, 8)
results = {}

for n_factors in factor_range:
    print(f"Running PMF with {n_factors} factors...")
    
    result = pmf.run_pmf(
        n_factors=n_factors,
        n_runs=20,  # Multiple runs for stability
        random_seed=42
    )
    
    results[n_factors] = result
    
    # Print model fit statistics
    print(f"Q/Qexp: {result.q_qexp:.2f}")
    print(f"Q robust: {result.q_robust:.2f}")
```

### 4. Selecting Optimal Number of Factors

```python
# Analyze Q/Qexp values to select optimal factor number
q_qexp_values = {n: results[n].q_qexp for n in factor_range}

print("Q/Qexp values by factor number:")
for n_factors, q_qexp in q_qexp_values.items():
    print(f"{n_factors} factors: {q_qexp:.2f}")

# Select the optimal number (typically where Q/Qexp stabilizes around 1)
optimal_factors = 5  # Example selection
optimal_result = results[optimal_factors]
```

### 5. Interpreting Results

```python
# Get factor profiles (source signatures)
factor_profiles = optimal_result.factor_profiles
print("\\nFactor Profiles (top species by contribution):")

for factor_idx in range(optimal_factors):
    print(f"\\nFactor {factor_idx + 1}:")
    # Sort species by their contribution to this factor
    factor_contributions = factor_profiles.iloc[:, factor_idx].sort_values(ascending=False)
    top_species = factor_contributions.head(5)
    
    for species, contribution in top_species.items():
        print(f"  {species}: {contribution:.2f}")

# Get factor contributions (time series)
factor_contributions = optimal_result.factor_contributions
print(f"\\nFactor contributions shape: {factor_contributions.shape}")
```

### 6. Visualization

```python
import matplotlib.pyplot as plt

# Create comprehensive visualizations
pmf.create_all_plots(
    result=optimal_result,
    dataset_name="Baltimore",
    output_dir="output"
)

# The following plots will be generated:
# - Factor profiles heatmap
# - Factor contributions heatmap  
# - Individual factor profile plots
# - Individual factor contribution plots
# - Top species by factor
# - Factor correlation analysis
```

## Expected Results

For the Baltimore dataset, you typically expect to identify factors representing:

1. **Traffic/Mobile Sources**: High in elemental carbon, organic carbon
2. **Secondary Sulfate**: Dominated by sulfate, ammonium
3. **Crustal/Dust**: High in aluminum, silicon, calcium
4. **Industrial Sources**: Metals like zinc, lead, copper
5. **Biomass Burning**: Potassium, organic carbon

## Interpreting Factor Profiles

- **High values** indicate species strongly associated with that source
- **Low values** suggest minimal contribution from that source
- **Chemical ratios** help identify source types (e.g., OC/EC ratios for traffic)

## Interpreting Factor Contributions

- **Time series patterns** reveal when sources are most active
- **Seasonal variations** may indicate temperature-dependent sources
- **Episodic peaks** could represent specific events or meteorological conditions

## Quality Assessment

Key indicators of a good PMF solution:

- **Q/Qexp ≈ 1**: Model fits the data appropriately
- **Stable factors**: Results consistent across multiple runs
- **Interpretable profiles**: Factors match known source signatures
- **Reasonable contributions**: Time series make physical sense

## Next Steps

After completing this analysis, you might want to:

1. **Sensitivity analysis**: Test different data preparation methods
2. **Bootstrap analysis**: Assess uncertainty in factor profiles
3. **Seasonal analysis**: Separate data by seasons
4. **Comparison with other sites**: Compare with [Baton Rouge](batch-processing.md) or St. Louis datasets

## Files Generated

This analysis creates several output files in the `output/` directory:

- `Baltimore_factor_profiles.csv`: Numerical factor profile data
- `Baltimore_factor_contributions.csv`: Time series contribution data
- `Baltimore_*.png`: Various visualization plots
- `Baltimore_summary.png`: Overview plot with key statistics

## Troubleshooting

**Common issues and solutions:**

- **High Q/Qexp values**: Try more factors or check data quality
- **Unstable factors**: Increase number of runs or check for outliers
- **Uninterpretable factors**: Consider different factor numbers or data preparation
- **Missing data warnings**: Review uncertainty treatment and missing value handling

For more advanced techniques, see [Custom Workflows](custom-workflows.md).
