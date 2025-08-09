
# Easy PMF

[![PyPI version](https://badge.fury.io/py/easy-pmf.svg)](https://badge.fury.io/py/easy-pmf)
[![Python versions](https://img.shields.io/pypi/pyversions/easy-pmf.svg)](https://pypi.org/project/easy-pmf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![CI/CD](https://github.com/gerritjandebruin/easy-pmf/actions/workflows/ci.yml/badge.svg)](https://github.com/gerritjandebruin/easy-pmf/actions/workflows/ci.yml)
[![Documentation](https://github.com/gerritjandebruin/easy-pmf/actions/workflows/docs.yml/badge.svg)](https://gerritjandebruin.github.io/easy-pmf/)
[![Publish](https://github.com/gerritjandebruin/easy-pmf/actions/workflows/publish.yml/badge.svg)](https://github.com/gerritjandebruin/easy-pmf/actions/workflows/publish.yml)

**Easy PMF** is a comprehensive Python package for Positive Matrix Factorization (PMF) analysis, designed specifically for environmental data analysis such as air quality source apportionment. It provides an easy-to-use interface similar to EPA's PMF software with built-in visualization capabilities.

:warning: This project is in the early stages of development and may not yet be suitable for production use.

:warning: A LLM (Large Language Model) is being used to assist with development and documentation; much of the content was vibe coded without much oversight.

## ✨ Features

- **Simple API**: Easy-to-use interface similar to scikit-learn
- **Comprehensive Visualizations**: EPA PMF-style plots and heatmaps
- **Multiple Dataset Support**: Built-in support for various environmental datasets
- **Robust Error Handling**: Input validation and convergence checking
- **Flexible Data Input**: Support for CSV, TXT, and Excel files
- **Interactive Analysis**: Command-line tools for quick analysis
- **Well Documented**: Extensive documentation with examples

## 🚀 Quick Start

### Installation

```bash
pip install easy-pmf
```

### Basic Usage

```python
import pandas as pd
from easy_pmf import PMF

# Load your concentration and uncertainty data
concentrations = pd.read_csv("concentrations.csv", index_col=0)
uncertainties = pd.read_csv("uncertainties.csv", index_col=0)

# Initialize PMF with 5 factors
pmf = PMF(n_components=5, random_state=42)

# Fit the model
pmf.fit(concentrations, uncertainties)

# Access results
factor_contributions = pmf.contributions_  # Time series of factor contributions
factor_profiles = pmf.profiles_            # Chemical profiles of each factor

# Check model performance
q_value = pmf.score(concentrations, uncertainties)
print(f"Model Q-value: {q_value:.2f}")
print(f"Converged: {pmf.converged_}")
print(f"Iterations: {pmf.n_iter_}")
```

### Command Line Interface

```bash
# Analyze a single dataset interactively
easy-pmf

# Or use the analysis scripts directly
python quick_analysis.py
```

## 📊 Included Example Datasets

The package comes with three real-world datasets:

- **Baton Rouge**: Air quality data (307 samples × 41 species)
- **St. Louis**: Environmental monitoring data (418 samples × 13 species)  
- **Baltimore**: PM2.5 composition data (657 samples × 26 species)

## 🎯 Use Cases

- **Air Quality Analysis**: Source apportionment of particulate matter
- **Environmental Monitoring**: Identifying pollution sources
- **Research**: Academic studies requiring PMF analysis
- **Regulatory Compliance**: EPA-style PMF analysis for reporting

## 📈 Visualization Capabilities

Easy PMF automatically generates comprehensive visualizations:

- **Factor Profiles**: Chemical signatures of each source
- **Factor Contributions**: Time series showing source strength
- **Correlation Matrices**: Relationships between factors
- **EPA-style Plots**: Publication-ready visualizations
- **Summary Dashboards**: Quick overview of results

## 📚 Documentation

### PMF Class Parameters

- `n_components` (int): Number of factors to extract
- `max_iter` (int, default=1000): Maximum iterations
- `tol` (float, default=1e-4): Convergence tolerance
- `random_state` (int, optional): Random seed for reproducibility

### Methods

- `fit(X, U=None)`: Fit PMF model to data
- `transform(X, U=None)`: Apply fitted model to new data
- `score(X, U=None)`: Calculate Q-value for goodness of fit

### Data Format Requirements

- **Concentrations**: Rows = time points, Columns = chemical species
- **Uncertainties**: Same format as concentrations (optional)
- **Index**: Date/time information
- **Values**: Non-negative concentrations

## 🛠️ Advanced Usage

### Custom Analysis Pipeline

```python
from easy_pmf import PMF
import matplotlib.pyplot as plt

# Load and preprocess data
concentrations = pd.read_csv("data.csv", index_col=0)
uncertainties = pd.read_csv("uncertainties.csv", index_col=0)

# Remove low-signal species
concentrations = concentrations.loc[:, (concentrations > 0).any(axis=0)]
uncertainties = uncertainties[concentrations.columns]

# Try different numbers of factors
for n_factors in range(3, 8):
    pmf = PMF(n_components=n_factors, random_state=42)
    pmf.fit(concentrations, uncertainties)
    q_value = pmf.score(concentrations, uncertainties)
    print(f"Factors: {n_factors}, Q-value: {q_value:.2f}")

# Analyze best model
best_pmf = PMF(n_components=5, random_state=42)
best_pmf.fit(concentrations, uncertainties)

# Custom visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot factor profiles
best_pmf.profiles_.T.plot(kind='bar', ax=ax1)
ax1.set_title('Factor Profiles')
ax1.set_xlabel('Chemical Species')

# Plot contributions over time
best_pmf.contributions_.plot(ax=ax2)
ax2.set_title('Factor Contributions Over Time')
ax2.set_ylabel('Contribution')

plt.tight_layout()
plt.show()
```

### Batch Processing Multiple Datasets

```python
import os
from easy_pmf import PMF

datasets = {
    "site1": {"conc": "site1_conc.csv", "unc": "site1_unc.csv"},
    "site2": {"conc": "site2_conc.csv", "unc": "site2_unc.csv"},
}

results = {}
for site, files in datasets.items():
    print(f"Analyzing {site}...")
    
    conc = pd.read_csv(files["conc"], index_col=0)
    unc = pd.read_csv(files["unc"], index_col=0)
    
    pmf = PMF(n_components=5, random_state=42)
    pmf.fit(conc, unc)
    
    results[site] = {
        "contributions": pmf.contributions_,
        "profiles": pmf.profiles_,
        "q_value": pmf.score(conc, unc),
        "converged": pmf.converged_
    }
    
    print(f"  Q-value: {results[site]['q_value']:.2f}")
    print(f"  Converged: {results[site]['converged']}")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/easy-pmf/easy-pmf.git
cd easy-pmf
pip install -e ".[dev]"
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EPA PMF software for inspiration
- Marloes van Os for first contributions and ideas

## 📞 Support

- **Documentation**: [https://easy-pmf.readthedocs.io](https://easy-pmf.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/easy-pmf/easy-pmf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/easy-pmf/easy-pmf/discussions)

---

**Easy PMF** - Making positive matrix factorization accessible to everyone! 🌍
