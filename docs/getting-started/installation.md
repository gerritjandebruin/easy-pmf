# Installation

## Prerequisites

Easy PMF requires Python 3.9 or later. We recommend using Python 3.10 or newer for the best experience.

## Installation Methods

### Via pip (Recommended)

The easiest way to install Easy PMF is using pip:

```bash
pip install easy-pmf
```

### Development Installation

If you want to contribute to Easy PMF or need the latest development version:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gerritjandebruin/easy-pmf.git
   cd easy-pmf
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

   This installs the package in "editable" mode with all development dependencies.

### Using uv (Alternative)

If you prefer using `uv` for Python package management:

```bash
uv add easy-pmf
```

For development:
```bash
git clone https://github.com/gerritjandebruin/easy-pmf.git
cd easy-pmf
uv sync
```

## Verify Installation

To verify that Easy PMF is correctly installed, run:

```python
import easy_pmf
print(easy_pmf.__version__)
```

Or test the command-line interface:

```bash
easy-pmf --version
```

## Dependencies

Easy PMF automatically installs the following core dependencies:

- **matplotlib** (≥3.5.0) - For creating visualizations
- **numpy** (≥1.20.0) - For numerical computations
- **pandas** (≥1.3.0) - For data manipulation
- **seaborn** (≥0.11.0) - For statistical visualizations

## Optional Dependencies

### Documentation

To build the documentation locally:

```bash
pip install easy-pmf[docs]
```

### Development

For development and testing:

```bash
pip install easy-pmf[dev]
```

This includes:

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **ruff** - Code linting and formatting
- **mypy** - Static type checking
- **build** - Package building
- **twine** - Package publishing

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors, ensure your Python environment is activated and Easy PMF is installed in the correct environment.

2. **Permission Errors**: On some systems, you might need to use `pip install --user easy-pmf` to install to your user directory.

3. **Version Conflicts**: If you have dependency conflicts, consider creating a new virtual environment:
   ```bash
   python -m venv easy_pmf_env
   source easy_pmf_env/bin/activate  # On Windows: easy_pmf_env\Scripts\activate
   pip install easy-pmf
   ```

### Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/gerritjandebruin/easy-pmf/issues) for similar problems
2. Create a new issue with your system information and error messages
3. Include your Python version (`python --version`) and operating system

## Next Steps

Once installed, proceed to the [Quick Start Guide](quick-start.md) to begin using Easy PMF!
