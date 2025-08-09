# Easy PMF Documentation

Welcome to the **Easy PMF** documentation! Easy PMF is a comprehensive Python package for Positive Matrix Factorization (PMF) analysis, designed specifically for environmental data analysis such as air quality source apportionment.

!!! warning "Development Status"
    This project is in the early stages of development and may not yet be suitable for production use. A Large Language Model (LLM) is being used to assist with development and documentation.

## ✨ Key Features

- **Simple API**: Easy-to-use interface similar to scikit-learn
- **Comprehensive Visualizations**: EPA PMF-style plots and heatmaps
- **Multiple Dataset Support**: Built-in support for various environmental datasets
- **Robust Error Handling**: Input validation and convergence checking
- **Flexible Data Input**: Support for CSV, TXT, and Excel files
- **Interactive Analysis**: Command-line tools for quick analysis
- **Well Documented**: Extensive documentation with examples

## 🚀 Quick Example

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
factor_contributions = pmf.contributions_
factor_profiles = pmf.profiles_

# Check model performance
q_value = pmf.score(concentrations, uncertainties)
print(f"Model Q-value: {q_value:.2f}")
```

## 📊 Use Cases

- **Air Quality Analysis**: Source apportionment of particulate matter
- **Environmental Monitoring**: Identifying pollution sources
- **Research**: Academic studies requiring PMF analysis
- **Regulatory Compliance**: EPA-style PMF analysis for reporting

## 📚 Getting Started

- [Installation Guide](getting-started/installation.md) - Install Easy PMF
- [Quick Start Tutorial](getting-started/quick-start.md) - Get up and running quickly
- [Basic Usage](getting-started/basic-usage.md) - Learn the fundamentals
- [API Reference](api/pmf.md) - Detailed API documentation

## 🎯 Example Datasets

The package comes with three real-world datasets ready for analysis:

- **Baton Rouge**: Air quality data (307 samples × 41 species)
- **St. Louis**: Environmental monitoring data (418 samples × 13 species)
- **Baltimore**: PM2.5 composition data (657 samples × 26 species)

## 🤝 Community

- **GitHub Repository**: [gerritjandebruin/easy-pmf](https://github.com/gerritjandebruin/easy-pmf)
- **PyPI Package**: [easy-pmf](https://pypi.org/project/easy-pmf/)
- **Issues**: [Report bugs or request features](https://github.com/gerritjandebruin/easy-pmf/issues)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/gerritjandebruin/easy-pmf/blob/main/LICENSE) file for details.
