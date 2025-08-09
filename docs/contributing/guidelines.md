# Contributing Guidelines

Thank you for your interest in contributing to Easy PMF! This document outlines the process for contributing code, documentation, bug reports, and feature requests.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Code](#contributing-code)
5. [Documentation](#documentation)
6. [Testing](#testing)
7. [Code Style](#code-style)
8. [Pull Request Process](#pull-request-process)
9. [Issue Reporting](#issue-reporting)
10. [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that ensures a welcoming environment for all contributors. Please be respectful and constructive in all interactions.

### Our Pledge

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Familiarity with PMF (Positive Matrix Factorization) concepts
- Understanding of atmospheric chemistry or environmental data analysis (helpful but not required)

### Development Environment

We use modern Python tooling for development:

- **uv**: For fast dependency management and virtual environments
- **ruff**: For code formatting and linting
- **pytest**: For testing
- **mypy**: For static type checking
- **mkdocs**: For documentation

## Development Setup

1. **Fork and Clone the Repository**

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/easy-pmf.git
cd easy-pmf
```

2. **Set Up Development Environment**

```bash
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install development dependencies
uv sync --all-extras
```

3. **Verify Installation**

```bash
# Run tests to ensure everything works
uv run pytest

# Check code style
uv run ruff check .

# Verify type checking
uv run mypy .
```

## Contributing Code

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality to the package
- **Performance improvements**: Optimize existing algorithms
- **Documentation**: Improve or add documentation
- **Examples**: Add new example analyses or datasets
- **Tests**: Improve test coverage

### Before You Start

1. **Check existing issues**: Look for related issues or feature requests
2. **Create an issue**: If none exists, create one describing your proposed changes
3. **Discuss the approach**: Get feedback before starting major work
4. **Check the roadmap**: Ensure your contribution aligns with project goals

### Development Workflow

1. **Create a Feature Branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

2. **Make Your Changes**

- Write clean, readable code
- Follow the established code style
- Add tests for new functionality
- Update documentation as needed

3. **Test Your Changes**

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=easy_pmf

# Check code style
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy .
```

4. **Commit Your Changes**

```bash
git add .
git commit -m "Add feature: brief description

Longer description of what was changed and why.
Fixes #issue-number"
```

5. **Push and Create Pull Request**

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples for new features
- Add docstrings to all public functions and classes
- Update relevant guides when adding features

### Documentation Types

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Step-by-step tutorials in `docs/user-guide/`
3. **Examples**: Complete analysis examples in `docs/examples/`
4. **Contributing Docs**: Development and contribution information

### Writing Docstrings

Use Google-style docstrings:

```python
def run_pmf(self, n_factors: int, n_runs: int = 20) -> PMFResult:
    """Run PMF analysis with specified parameters.

    This method performs Positive Matrix Factorization on the prepared
    dataset using the specified number of factors.

    Args:
        n_factors: Number of factors to extract (typically 3-10)
        n_runs: Number of independent runs for stability (default: 20)

    Returns:
        PMFResult object containing factor profiles, contributions,
        and model fit statistics.

    Raises:
        ValueError: If n_factors is less than 2 or greater than number of species
        DataError: If data has not been prepared before running PMF

    Example:
        >>> pmf = PMF()
        >>> pmf.load_data("concentrations.csv", "uncertainties.csv")
        >>> pmf.prepare_data()
        >>> result = pmf.run_pmf(n_factors=5, n_runs=20)
        >>> print(f"Q/Qexp: {result.q_qexp:.2f}")
    """
```

### Building Documentation

```bash
# Build documentation locally
uv run mkdocs build

# Serve documentation for development
uv run mkdocs serve
```

## Testing

### Test Philosophy

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Example tests**: Ensure examples in documentation work

### Writing Tests

1. **Test File Organization**

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── examples/       # Tests for example code
└── data/          # Test datasets
```

2. **Test Naming Convention**

```python
def test_function_name_expected_behavior():
    """Test that function_name does expected_behavior when given specific input."""
```

3. **Test Structure**

```python
def test_pmf_run_with_valid_data():
    """Test that PMF runs successfully with valid input data."""
    # Arrange
    pmf = PMF()
    pmf.load_test_data()
    pmf.prepare_data()

    # Act
    result = pmf.run_pmf(n_factors=3)

    # Assert
    assert result.q_qexp > 0
    assert result.factor_profiles.shape == (pmf.n_species, 3)
    assert result.factor_contributions.shape == (pmf.n_samples, 3)
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_core.py

# Run with coverage
uv run pytest --cov=easy_pmf --cov-report=html

# Run tests and open coverage report
uv run pytest --cov=easy_pmf --cov-report=html
# Then open htmlcov/index.html
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: isort compatible
- **Docstrings**: Google style
- **Type hints**: Required for all public functions

### Automated Formatting

```bash
# Format code automatically
uv run ruff format .

# Check for style issues
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Code Quality Standards

1. **Type Hints**

```python
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

def process_data(
    data: pd.DataFrame,
    species_list: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Process input data and return processed results."""
```

2. **Error Handling**

```python
def load_data(self, filepath: str) -> None:
    """Load data from file with proper error handling."""
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {filepath}")

    if data.empty:
        raise ValueError("Loaded data is empty")
```

3. **Logging**

```python
import logging

logger = logging.getLogger(__name__)

def run_analysis(self):
    """Run analysis with appropriate logging."""
    logger.info("Starting PMF analysis")
    try:
        result = self._perform_calculation()
        logger.info(f"Analysis completed successfully: Q/Qexp = {result.q_qexp:.3f}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Check code style** compliance
5. **Update CHANGELOG.md** if applicable

### Pull Request Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] Examples updated if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Breaking changes documented
```

### Review Process

1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer review required
3. **Testing**: Verify tests cover new functionality
4. **Documentation**: Ensure documentation is complete

### Merge Requirements

- All CI checks passing
- At least one approving review from maintainer
- No requested changes outstanding
- Branch up to date with main

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Clear title** describing the issue
2. **Steps to reproduce** the problem
3. **Expected vs actual behavior**
4. **Environment information** (Python version, OS, package version)
5. **Minimal example** that demonstrates the issue
6. **Error messages** and stack traces

### Feature Requests

When requesting features, include:

1. **Clear description** of the proposed feature
2. **Use case** explaining why it's needed
3. **Proposed implementation** if you have ideas
4. **Examples** of how it would be used

### Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release notes
4. Tag the release
5. Build and publish to PyPI
6. Update documentation

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Documentation**: Comprehensive guides and API reference

### Mentorship

New contributors are welcome! If you're new to open source:

1. Look for issues labeled `good first issue`
2. Ask questions in GitHub Discussions
3. Start with documentation improvements
4. Reach out to maintainers for guidance

## Recognition

Contributors are recognized in:

- **CHANGELOG.md**: Major contributions noted in release notes
- **README.md**: Contributors section
- **GitHub**: Contributor graphs and statistics

Thank you for contributing to Easy PMF! Your contributions help make atmospheric data analysis more accessible to researchers worldwide.
