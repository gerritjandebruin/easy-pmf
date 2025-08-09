# Quick Start

This guide will get you up and running with Easy PMF in just a few minutes!

## 5-Minute Tutorial

### Step 1: Import Easy PMF

```python
import pandas as pd
from easy_pmf import PMF
```

### Step 2: Load Your Data

Easy PMF works with pandas DataFrames. You need concentration data and optionally uncertainty data:

```python
# Load concentration data (required)
concentrations = pd.read_csv("concentrations.csv", index_col=0)

# Load uncertainty data (optional but recommended)
uncertainties = pd.read_csv("uncertainties.csv", index_col=0)
```

!!! tip "Data Format"
    - **Rows**: Time points or sample dates
    - **Columns**: Chemical species or pollutants
    - **Values**: Non-negative concentrations
    - **Index**: Preferably datetime for time series analysis

### Step 3: Create and Fit the Model

```python
# Initialize PMF model with 5 factors
pmf = PMF(n_components=5, random_state=42)

# Fit the model to your data
pmf.fit(concentrations, uncertainties)
```

### Step 4: Access Results

```python
# Get factor contributions (time series)
contributions = pmf.contributions_
print("Factor Contributions Shape:", contributions.shape)

# Get factor profiles (chemical signatures)
profiles = pmf.profiles_
print("Factor Profiles Shape:", profiles.shape)

# Check model quality
q_value = pmf.score(concentrations, uncertainties)
print(f"Q-value: {q_value:.2f}")
print(f"Converged: {pmf.converged_}")
print(f"Iterations: {pmf.n_iter_}")
```

## Using Built-in Datasets

Easy PMF comes with example datasets you can use immediately:

### Interactive Analysis

Run the interactive command-line tool:

```bash
easy-pmf
```

This will guide you through analyzing the built-in datasets.

### Programmatic Access

```python
import pandas as pd
from pathlib import Path

# Assuming you have the example data files
data_dir = Path("data")

# Baltimore dataset example
conc_file = data_dir / "Dataset-Baltimore_con.txt"
unc_file = data_dir / "Dataset-Baltimore_unc.txt"

# Load the data
concentrations = pd.read_csv(conc_file, sep='\t', index_col=0)
uncertainties = pd.read_csv(unc_file, sep='\t', index_col=0)

# Run PMF analysis
pmf = PMF(n_components=7, random_state=42)
pmf.fit(concentrations, uncertainties)

print(f"Analysis complete! Q-value: {pmf.score(concentrations, uncertainties):.2f}")
```

## Command Line Interface

Easy PMF provides a convenient CLI for quick analysis:

### Interactive Mode (Default)

```bash
easy-pmf --interactive
```

### Batch Analysis

```bash
easy-pmf --analyze-all
```

### Custom Parameters

```bash
easy-pmf --factors 5 --data-dir ./my_data --output-dir ./results
```

### CLI Options

- `--interactive, -i`: Run interactive analysis (default)
- `--analyze-all, -a`: Analyze all datasets automatically
- `--data-dir`: Directory containing data files (default: data)
- `--output-dir`: Directory for output files (default: output)
- `--factors, -f`: Number of PMF factors (default: 7)
- `--version, -v`: Show version information

## Basic Visualization

While detailed visualization is covered in the [User Guide](../user-guide/visualization.md), here's a quick preview:

```python
import matplotlib.pyplot as plt

# Plot factor contributions over time
pmf.contributions_.plot(figsize=(12, 6))
plt.title('Factor Contributions Over Time')
plt.ylabel('Contribution')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot factor profiles as a heatmap
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(pmf.profiles_, annot=True, fmt='.2f', cmap='viridis')
plt.title('Factor Profiles (Chemical Signatures)')
plt.tight_layout()
plt.show()
```

## What's Next?

Now that you've got Easy PMF running, explore these topics:

1. **[Basic Usage](basic-usage.md)** - Learn the fundamentals of PMF analysis
2. **[Data Preparation](../user-guide/data-preparation.md)** - Prepare your own datasets
3. **[Interpreting Results](../user-guide/interpreting-results.md)** - Understand your PMF results
4. **[Examples](../examples/datasets.md)** - Explore detailed examples with real data

## Troubleshooting

### Common Quick Start Issues

**Problem**: Import error
```python
ImportError: No module named 'easy_pmf'
```
**Solution**: Ensure Easy PMF is installed: `pip install easy-pmf`

**Problem**: Data format error
```python
ValueError: x contains negative values
```
**Solution**: PMF requires non-negative data. Check your concentration values.

**Problem**: Shape mismatch
```python
ValueError: x and u must have the same shape
```
**Solution**: Ensure concentration and uncertainty DataFrames have identical dimensions.

**Problem**: Convergence warning
```python
UserWarning: PMF did not converge after 1000 iterations
```
**Solution**: Increase `max_iter` or adjust `tol` parameters, or try different `n_components`.

Need more help? Check the [User Guide](../user-guide/pmf-basics.md) or [API Reference](../api/pmf.md)!
