# Development Setup

This guide will help you set up a development environment for contributing to Easy PMF.

## Prerequisites

- Python 3.9 or later
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setting Up Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/gerritjandebruin/easy-pmf.git
cd easy-pmf
```

### 2. Set Up Python Environment

#### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment with dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Using pip and venv

```bash
# Create virtual environment
python -m venv easy_pmf_env
source easy_pmf_env/bin/activate  # On Windows: easy_pmf_env\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests to verify everything works
uv run pytest

# Check code style
uv run ruff check

# Run type checking
uv run mypy .

# Test CLI
easy-pmf --version
```

## Development Workflow

### 1. Code Quality Tools

Easy PMF uses several tools to maintain code quality:

#### Ruff (Linting and Formatting)

```bash
# Check for linting issues
uv run ruff check

# Fix auto-fixable issues
uv run ruff check --fix

# Format code
uv run ruff format
```

#### MyPy (Type Checking)

```bash
# Run type checking
uv run mypy .

# Run type checking on specific file
uv run mypy src/easy_pmf/pmf.py
```

#### Pre-commit Hooks (Optional)

Set up pre-commit hooks to automatically check code before commits:

```bash
pip install pre-commit
pre-commit install
```

### 2. Testing

#### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=easy_pmf --cov-report=html

# Run specific test file
uv run pytest tests/test_pmf.py

# Run specific test
uv run pytest tests/test_pmf.py::test_pmf_basic_functionality
```

#### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
import pandas as pd
from easy_pmf import PMF

def test_new_feature():
    """Test description."""
    # Create test data
    concentrations = pd.DataFrame({
        'Species1': [1.0, 2.0, 3.0],
        'Species2': [2.0, 4.0, 6.0]
    })

    # Test your feature
    pmf = PMF(n_components=2)
    pmf.fit(concentrations)

    # Assertions
    assert pmf.converged_
    assert pmf.contributions_.shape == (3, 2)
```

### 3. Documentation

#### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
# Open http://127.0.0.1:8000 in your browser
```

#### Writing Documentation

- Use Google-style docstrings in code
- Add examples to docstrings
- Update relevant `.md` files in `docs/`
- Include type hints

Example docstring:

```python
def new_function(data: pd.DataFrame, n_components: int = 5) -> PMF:
    """Short description of the function.

    Longer description explaining what the function does,
    its use cases, and any important details.

    Args:
        data: Input concentration data with samples as rows and species as columns.
        n_components: Number of PMF factors to extract. Defaults to 5.

    Returns:
        Fitted PMF model instance.

    Raises:
        ValueError: If data contains negative values.

    Examples:
        Basic usage:

        >>> import pandas as pd
        >>> from easy_pmf import PMF
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 4, 6]})
        >>> pmf = new_function(data, n_components=2)
        >>> pmf.converged_
        True
    """
```

## Project Structure

```
easy-pmf/
├── src/
│   └── easy_pmf/
│       ├── __init__.py          # Main PMF class
│       └── cli.py               # Command-line interface
├── tests/
│   ├── test_pmf.py             # PMF class tests
│   └── test_cli.py             # CLI tests
├── docs/
│   ├── index.md                # Documentation home
│   ├── getting-started/        # Getting started guides
│   ├── user-guide/            # User guides
│   ├── api/                   # API reference
│   └── examples/              # Examples
├── data/                      # Example datasets
├── pyproject.toml             # Project configuration
├── mkdocs.yml                # Documentation configuration
├── README.md                 # Project overview
└── CHANGELOG.md             # Version history
```

## Adding New Features

### 1. Planning

Before implementing:

1. **Check existing issues** on GitHub
2. **Open a discussion** for major features
3. **Create an issue** describing the feature
4. **Get feedback** from maintainers

### 2. Implementation Steps

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement the feature**:
   - Write code following existing patterns
   - Add type hints
   - Include docstrings
   - Handle edge cases

3. **Add tests**:
   - Test normal usage
   - Test edge cases
   - Test error conditions
   - Aim for >90% coverage

4. **Update documentation**:
   - Add docstrings to new functions/classes
   - Update relevant documentation files
   - Add examples if applicable

5. **Test everything**:
   ```bash
   uv run pytest
   uv run ruff check
   uv run mypy .
   ```

### 3. Example: Adding a New Method

```python
# In src/easy_pmf/__init__.py

class PMF:
    # ... existing methods ...

    def validate_results(self, threshold: float = 0.1) -> Dict[str, bool]:
        """Validate PMF results using various criteria.

        Args:
            threshold: Validation threshold for various checks.

        Returns:
            Dictionary with validation results.

        Examples:
            >>> pmf = PMF(n_components=5)
            >>> pmf.fit(concentrations, uncertainties)
            >>> validation = pmf.validate_results()
            >>> validation['converged']
            True
        """
        if self.contributions_ is None:
            raise ValueError("Model must be fitted before validation")

        validation = {
            'converged': self.converged_,
            'reasonable_contributions': (self.contributions_ >= 0).all().all(),
            'reasonable_profiles': (self.profiles_ >= 0).all().all(),
        }

        return validation
```

## Development Guidelines

### Code Style

1. **Follow PEP 8** with line length of 88 characters
2. **Use type hints** for all function parameters and returns
3. **Write descriptive variable names**
4. **Keep functions focused** and reasonably sized
5. **Use docstrings** for all public functions and classes

### Git Workflow

1. **Use descriptive commit messages**:
   ```bash
   git commit -m "Add validation method to PMF class

   - Implement validate_results method
   - Add tests for validation functionality
   - Update documentation with examples"
   ```

2. **Keep commits focused** on single changes
3. **Rebase** before submitting pull requests
4. **Update CHANGELOG.md** for significant changes

### Testing Guidelines

1. **Test all public APIs**
2. **Include edge cases**
3. **Test error conditions**
4. **Use meaningful test names**
5. **Keep tests independent**

### Documentation Guidelines

1. **Use Google-style docstrings**
2. **Include examples** in docstrings
3. **Update user guides** for new features
4. **Add API documentation** for new classes/functions
5. **Keep documentation** up to date

## Common Development Tasks

### Adding a New Dataset

1. **Add data files** to `data/` directory
2. **Update example documentation**
3. **Add tests** for loading the dataset
4. **Update CLI** if needed

### Improving Algorithm Performance

1. **Profile current performance**
2. **Implement improvements**
3. **Add benchmarks**
4. **Ensure backward compatibility**

### Adding Visualization Features

1. **Design API** for new plots
2. **Implement plotting functions**
3. **Add to visualization guide**
4. **Include examples**

## Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Code Review**: Submit pull requests for feedback
- **Documentation**: Check existing docs and examples

## Next Steps

- Read the [Contributing Guidelines](guidelines.md)
- Learn about [Testing](testing.md) practices
- Check out existing [Issues](https://github.com/gerritjandebruin/easy-pmf/issues)
- Join the community discussions
