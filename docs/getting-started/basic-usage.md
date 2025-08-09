# Basic Usage

This guide covers the fundamental concepts and usage patterns of Easy PMF.

## Core Concepts

### What is PMF?

Positive Matrix Factorization (PMF) is a multivariate factor analysis tool that decomposes a matrix of speciated sample data into factor contributions and factor profiles:

**X = G × F + E**

Where:
- **X**: Original data matrix (samples × species)
- **G**: Factor contributions matrix (samples × factors)
- **F**: Factor profiles matrix (factors × species)
- **E**: Residual matrix (unexplained variance)

### Key Components

1. **Factor Contributions (G matrix)**: How much each source contributes to each sample over time
2. **Factor Profiles (F matrix)**: The chemical "fingerprint" or signature of each source
3. **Q-value**: Goodness-of-fit metric (lower values indicate better fit)

## The PMF Class

### Initialization

```python
from easy_pmf import PMF

# Basic initialization
pmf = PMF(n_components=5)

# With all parameters
pmf = PMF(
    n_components=5,        # Number of factors/sources
    max_iter=1000,         # Maximum iterations
    tol=1e-4,              # Convergence tolerance
    random_state=42        # For reproducible results
)
```

### Parameters Explained

- **`n_components`**: The number of factors (sources) to extract. This is often determined through trial-and-error or domain knowledge.
- **`max_iter`**: Maximum number of iterations for the algorithm to converge.
- **`tol`**: Convergence tolerance. Algorithm stops when changes between iterations are smaller than this value.
- **`random_state`**: Seed for random number generation to ensure reproducible results.

## Data Requirements

### Input Data Format

Easy PMF expects pandas DataFrames with specific structure:

```python
import pandas as pd

# Concentrations DataFrame
concentrations = pd.DataFrame({
    'PM2.5': [25.3, 18.7, 32.1, ...],
    'NO3': [5.2, 3.8, 7.1, ...],
    'SO4': [8.1, 6.3, 9.8, ...],
    # ... more species
}, index=pd.date_range('2023-01-01', periods=100, freq='D'))

# Uncertainties DataFrame (same structure)
uncertainties = pd.DataFrame({
    'PM2.5': [2.5, 1.9, 3.2, ...],
    'NO3': [0.5, 0.4, 0.7, ...],
    'SO4': [0.8, 0.6, 1.0, ...],
    # ... more species
}, index=concentrations.index)
```

### Data Quality Requirements

!!! warning "Important Data Requirements"
    - **Non-negative values**: All concentrations must be ≥ 0
    - **No missing values**: Handle NaN values before fitting
    - **Consistent dimensions**: Concentration and uncertainty matrices must have the same shape
    - **Positive uncertainties**: Uncertainty values must be > 0

### Data Preprocessing

```python
# Remove negative values
concentrations = concentrations.clip(lower=0)

# Handle missing values
concentrations = concentrations.fillna(concentrations.median())
uncertainties = uncertainties.fillna(uncertainties.median())

# Remove species with too many zeros
min_detection_rate = 0.5  # 50% detection rate
valid_species = (concentrations > 0).mean() >= min_detection_rate
concentrations = concentrations.loc[:, valid_species]
uncertainties = uncertainties.loc[:, valid_species]
```

## Basic Workflow

### 1. Load and Prepare Data

```python
import pandas as pd
from easy_pmf import PMF

# Load your data
concentrations = pd.read_csv("data.csv", index_col=0)
uncertainties = pd.read_csv("uncertainties.csv", index_col=0)

# Basic validation
print(f"Data shape: {concentrations.shape}")
print(f"Date range: {concentrations.index.min()} to {concentrations.index.max()}")
print(f"Species: {list(concentrations.columns)}")
```

### 2. Initialize and Fit Model

```python
# Create PMF instance
pmf = PMF(n_components=5, random_state=42)

# Fit the model
pmf.fit(concentrations, uncertainties)

# Check convergence
print(f"Converged: {pmf.converged_}")
print(f"Iterations: {pmf.n_iter_}")
```

### 3. Evaluate Model Quality

```python
# Calculate Q-value
q_value = pmf.score(concentrations, uncertainties)
print(f"Q-value: {q_value:.2f}")

# Check factor contributions
print("Factor contributions summary:")
print(pmf.contributions_.describe())

# Check factor profiles
print("Factor profiles summary:")
print(pmf.profiles_.describe())
```

### 4. Access Results

```python
# Factor contributions (time series)
contributions = pmf.contributions_
print("Contributions shape:", contributions.shape)
print(contributions.head())

# Factor profiles (chemical signatures)
profiles = pmf.profiles_
print("Profiles shape:", profiles.shape)
print(profiles.head())
```

## Model Selection

### Choosing the Number of Factors

```python
import matplotlib.pyplot as plt

# Test different numbers of factors
n_factors_range = range(3, 10)
q_values = []

for n in n_factors_range:
    pmf_test = PMF(n_components=n, random_state=42)
    pmf_test.fit(concentrations, uncertainties)
    q_val = pmf_test.score(concentrations, uncertainties)
    q_values.append(q_val)
    print(f"Factors: {n}, Q-value: {q_val:.2f}, Converged: {pmf_test.converged_}")

# Plot Q-values vs number of factors
plt.figure(figsize=(8, 6))
plt.plot(n_factors_range, q_values, 'bo-')
plt.xlabel('Number of Factors')
plt.ylabel('Q-value')
plt.title('Model Selection: Q-value vs Number of Factors')
plt.grid(True, alpha=0.3)
plt.show()
```

### Convergence Diagnostics

```python
# Check convergence history if available
if hasattr(pmf, '_convergence_history'):
    plt.figure(figsize=(10, 6))
    plt.semilogy(pmf._convergence_history)
    plt.xlabel('Iteration')
    plt.ylabel('Convergence Metric (log scale)')
    plt.title('PMF Convergence History')
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Advanced Usage

### Transform New Data

```python
# Apply fitted model to new data
new_concentrations = pd.read_csv("new_data.csv", index_col=0)
new_contributions = pmf.transform(new_concentrations)

print("New contributions shape:", new_contributions.shape)
print(new_contributions.head())
```

### Batch Processing

```python
# Process multiple datasets
datasets = {
    'site1': 'site1_data.csv',
    'site2': 'site2_data.csv',
    'site3': 'site3_data.csv'
}

results = {}
for site, filename in datasets.items():
    print(f"Processing {site}...")

    # Load data
    data = pd.read_csv(filename, index_col=0)

    # Fit PMF
    pmf = PMF(n_components=5, random_state=42)
    pmf.fit(data)

    # Store results
    results[site] = {
        'contributions': pmf.contributions_,
        'profiles': pmf.profiles_,
        'q_value': pmf.score(data),
        'converged': pmf.converged_
    }

    print(f"  Q-value: {results[site]['q_value']:.2f}")
    print(f"  Converged: {results[site]['converged']}")
```

## Error Handling

### Common Errors and Solutions

```python
try:
    pmf = PMF(n_components=5)
    pmf.fit(concentrations, uncertainties)
except ValueError as e:
    if "negative values" in str(e):
        print("Fix negative values in data")
        concentrations = concentrations.clip(lower=0)
    elif "NaN values" in str(e):
        print("Handle missing values")
        concentrations = concentrations.fillna(method='forward')
    elif "different shapes" in str(e):
        print("Ensure concentration and uncertainty matrices have same shape")
    else:
        print(f"Unexpected error: {e}")
```

### Validation Utilities

```python
def validate_pmf_data(concentrations, uncertainties=None):
    """Validate data for PMF analysis."""
    issues = []

    # Check for negative values
    if (concentrations < 0).any().any():
        issues.append("Negative concentrations found")

    # Check for NaN values
    if concentrations.isnull().any().any():
        issues.append("Missing values in concentrations")

    # Check uncertainties if provided
    if uncertainties is not None:
        if concentrations.shape != uncertainties.shape:
            issues.append("Shape mismatch between concentrations and uncertainties")
        if (uncertainties <= 0).any().any():
            issues.append("Non-positive uncertainties found")

    # Check for sufficient data
    if concentrations.shape[0] < 50:
        issues.append("Recommend at least 50 samples for stable PMF")

    if concentrations.shape[1] < 5:
        issues.append("Recommend at least 5 species for meaningful PMF")

    return issues

# Use the validator
issues = validate_pmf_data(concentrations, uncertainties)
if issues:
    print("Data validation issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Data validation passed!")
```

## Next Steps

- Learn about [Data Preparation](../user-guide/data-preparation.md) techniques
- Explore [Result Interpretation](../user-guide/interpreting-results.md)
- Check out detailed [Examples](../examples/datasets.md)
- Review the complete [API Reference](../api/pmf.md)
