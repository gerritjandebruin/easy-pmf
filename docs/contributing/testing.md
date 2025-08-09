# Testing Guide

This guide covers testing procedures for Easy PMF, including how to run tests, write new tests, and ensure code quality.

## Overview

Easy PMF uses a comprehensive testing strategy to ensure reliability:

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test component interactions  
- **End-to-end tests**: Test complete analysis workflows
- **Documentation tests**: Ensure examples work correctly
- **Performance tests**: Monitor algorithm performance

## Testing Framework

### Tools Used

- **pytest**: Primary testing framework
- **pytest-cov**: Code coverage measurement
- **pytest-xdist**: Parallel test execution
- **pytest-mock**: Mocking capabilities
- **hypothesis**: Property-based testing

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_core.py        # Core PMF functionality
│   ├── test_data_loading.py # Data loading and validation
│   ├── test_preprocessing.py # Data preprocessing
│   └── test_visualization.py # Plotting functions
├── integration/             # Integration tests
│   ├── test_workflows.py   # End-to-end workflows
│   └── test_api.py         # Public API integration
├── examples/               # Documentation example tests
│   ├── test_baltimore.py   # Baltimore example tests
│   └── test_batch_processing.py # Batch processing tests
├── performance/            # Performance benchmarks
│   └── test_benchmarks.py
└── data/                   # Test datasets
    ├── small_dataset.csv
    └── test_uncertainties.csv
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_core.py

# Run specific test function
uv run pytest tests/unit/test_core.py::test_pmf_initialization

# Run tests matching pattern
uv run pytest -k "test_load"
```

### Test Coverage

```bash
# Run tests with coverage
uv run pytest --cov=easy_pmf

# Generate HTML coverage report
uv run pytest --cov=easy_pmf --cov-report=html

# Generate detailed coverage report
uv run pytest --cov=easy_pmf --cov-report=term-missing

# Set minimum coverage threshold
uv run pytest --cov=easy_pmf --cov-fail-under=90
```

### Parallel Testing

```bash
# Run tests in parallel (faster for large test suites)
uv run pytest -n auto

# Run with specific number of workers
uv run pytest -n 4
```

### Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests  
uv run pytest tests/integration/

# Run quick tests (skip slow ones)
uv run pytest -m "not slow"

# Run only slow tests
uv run pytest -m slow
```

## Writing Tests

### Test Structure

Use the **Arrange-Act-Assert** pattern:

```python
def test_pmf_run_with_valid_data():
    """Test that PMF runs successfully with valid input data."""
    # Arrange: Set up test data and objects
    pmf = PMF()
    pmf.concentration_data = create_test_concentration_data()
    pmf.uncertainty_data = create_test_uncertainty_data()
    
    # Act: Perform the operation being tested
    result = pmf.run_pmf(n_factors=3, n_runs=5)
    
    # Assert: Check the results
    assert isinstance(result, PMFResult)
    assert result.factor_profiles.shape == (pmf.n_species, 3)
    assert result.factor_contributions.shape == (pmf.n_samples, 3)
    assert result.q_qexp > 0
```

### Test Fixtures

Use fixtures for reusable test data:

```python
# In conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_concentration_data():
    """Create sample concentration data for testing."""
    np.random.seed(42)  # Reproducible random data
    data = np.random.lognormal(mean=1, sigma=1, size=(100, 10))
    species = [f'Species_{i}' for i in range(10)]
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame(data, index=dates, columns=species)

@pytest.fixture
def sample_uncertainty_data(sample_concentration_data):
    """Create uncertainty data matching concentration data."""
    # 10% relative uncertainty
    uncertainty = sample_concentration_data * 0.1
    return uncertainty

@pytest.fixture
def pmf_with_data(sample_concentration_data, sample_uncertainty_data):
    """Create PMF instance with test data loaded."""
    pmf = PMF()
    pmf.concentration_data = sample_concentration_data
    pmf.uncertainty_data = sample_uncertainty_data
    pmf.prepare_data()
    return pmf

# Using fixtures in tests
def test_pmf_analysis(pmf_with_data):
    """Test complete PMF analysis workflow."""
    result = pmf_with_data.run_pmf(n_factors=3)
    assert result.q_qexp > 0
```

### Parameterized Tests

Test multiple scenarios efficiently:

```python
import pytest

@pytest.mark.parametrize("n_factors,expected_shape", [
    (2, (10, 2)),
    (3, (10, 3)),
    (5, (10, 5)),
])
def test_factor_profiles_shape(pmf_with_data, n_factors, expected_shape):
    """Test that factor profiles have correct shape for different factor numbers."""
    result = pmf_with_data.run_pmf(n_factors=n_factors)
    assert result.factor_profiles.shape == expected_shape

@pytest.mark.parametrize("file_format,separator", [
    ("csv", ","),
    ("txt", "\\t"),
    ("tsv", "\\t"),
])
def test_data_loading_formats(tmp_path, file_format, separator):
    """Test loading data in different file formats."""
    # Create test file
    data = create_test_data()
    file_path = tmp_path / f"test_data.{file_format}"
    data.to_csv(file_path, sep=separator)
    
    # Test loading
    pmf = PMF()
    pmf.load_data(str(file_path))
    assert pmf.concentration_data.shape == data.shape
```

### Exception Testing

Test error conditions:

```python
def test_pmf_run_without_data():
    """Test that PMF raises error when run without data."""
    pmf = PMF()
    
    with pytest.raises(ValueError, match="No data loaded"):
        pmf.run_pmf(n_factors=3)

def test_invalid_factor_number(pmf_with_data):
    """Test that invalid factor numbers raise appropriate errors."""
    # Too few factors
    with pytest.raises(ValueError, match="n_factors must be at least 2"):
        pmf_with_data.run_pmf(n_factors=1)
    
    # Too many factors
    n_species = pmf_with_data.concentration_data.shape[1]
    with pytest.raises(ValueError, match="n_factors cannot exceed"):
        pmf_with_data.run_pmf(n_factors=n_species + 1)

def test_file_not_found():
    """Test handling of missing data files."""
    pmf = PMF()
    
    with pytest.raises(FileNotFoundError):
        pmf.load_data("nonexistent_file.csv")
```

### Mocking External Dependencies

Mock external dependencies for isolated testing:

```python
from unittest.mock import patch, MagicMock

def test_data_loading_with_network_failure():
    """Test graceful handling of network failures."""
    pmf = PMF()
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = ConnectionError("Network unavailable")
        
        with pytest.raises(ConnectionError):
            pmf.load_data("http://example.com/data.csv")

@patch('matplotlib.pyplot.savefig')
def test_plot_generation(mock_savefig, pmf_with_data):
    """Test that plots are generated without actually saving files."""
    result = pmf_with_data.run_pmf(n_factors=3)
    pmf_with_data.create_plots(result, output_dir="test_output")
    
    # Verify savefig was called
    assert mock_savefig.called
    assert mock_savefig.call_count > 0
```

## Test Categories and Markers

### Custom Test Markers

Define custom markers in `pytest.ini`:

```ini
[tool:pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    requires_data: marks tests that need external data files
    visualization: marks tests that generate plots
```

Use markers in tests:

```python
import pytest

@pytest.mark.slow
def test_large_dataset_analysis():
    """Test analysis with large dataset (takes >10 seconds)."""
    # This test will be skipped when running with -m "not slow"
    pass

@pytest.mark.requires_data
def test_real_dataset_analysis():
    """Test with real atmospheric data."""
    # Only run if external data is available
    pass

@pytest.mark.visualization
def test_plot_generation():
    """Test plot generation functions."""
    pass
```

### Conditional Tests

Skip tests based on conditions:

```python
import sys
import pytest

@pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
def test_unix_specific_feature():
    """Test that only runs on Unix systems."""
    pass

@pytest.mark.skipif(not pytest.importorskip("matplotlib"),
                   reason="matplotlib not available")
def test_plotting_function():
    """Test that requires matplotlib."""
    pass

def test_optional_dependency():
    """Test with optional dependency."""
    try:
        import seaborn
    except ImportError:
        pytest.skip("seaborn not available")
    
    # Test code using seaborn
    pass
```

## Performance Testing

### Benchmark Tests

Monitor performance over time:

```python
import time
import pytest

class TestPerformance:
    """Performance benchmark tests."""
    
    def test_pmf_performance_small_dataset(self, benchmark, small_dataset):
        """Benchmark PMF performance on small dataset."""
        pmf = PMF()
        pmf.concentration_data = small_dataset
        pmf.prepare_data()
        
        # Benchmark the PMF run
        result = benchmark(pmf.run_pmf, n_factors=3, n_runs=5)
        assert result.q_qexp > 0
    
    @pytest.mark.slow
    def test_pmf_performance_large_dataset(self, benchmark, large_dataset):
        """Benchmark PMF performance on large dataset."""
        pmf = PMF()
        pmf.concentration_data = large_dataset
        pmf.prepare_data()
        
        result = benchmark(pmf.run_pmf, n_factors=5, n_runs=10)
        assert result.q_qexp > 0
    
    def test_memory_usage(self, memory_profiler, sample_data):
        """Test memory usage during PMF analysis."""
        pmf = PMF()
        pmf.concentration_data = sample_data
        pmf.prepare_data()
        
        # Monitor memory during execution
        with memory_profiler:
            result = pmf.run_pmf(n_factors=3)
        
        # Assert memory usage is reasonable
        assert memory_profiler.max_memory < 1000  # MB
```

### Load Testing

Test with various data sizes:

```python
@pytest.mark.parametrize("n_samples,n_species", [
    (50, 10),      # Small dataset
    (500, 20),     # Medium dataset
    (2000, 50),    # Large dataset
])
def test_scalability(n_samples, n_species):
    """Test PMF scalability with different data sizes."""
    # Generate synthetic data
    data = generate_synthetic_data(n_samples, n_species)
    uncertainties = data * 0.1
    
    pmf = PMF()
    pmf.concentration_data = data
    pmf.uncertainty_data = uncertainties
    pmf.prepare_data()
    
    start_time = time.time()
    result = pmf.run_pmf(n_factors=min(5, n_species-1))
    execution_time = time.time() - start_time
    
    # Performance assertions
    assert result.q_qexp > 0
    assert execution_time < 60  # Should complete within 1 minute
```

## Continuous Integration

### GitHub Actions Workflow

Example `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install uv
      run: pip install uv
    
    - name: Install dependencies
      run: uv sync --all-extras
    
    - name: Run tests
      run: uv run pytest --cov=easy_pmf --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Quality Gates

Set minimum standards:

```bash
# In CI/CD pipeline or pre-commit hooks

# Minimum test coverage
uv run pytest --cov=easy_pmf --cov-fail-under=85

# Code style checks
uv run ruff check .
uv run ruff format --check .

# Type checking
uv run mypy .

# Security checks
uv run safety check

# Documentation builds
uv run mkdocs build --strict
```

## Test Data Management

### Test Datasets

Organize test data efficiently:

```python
# tests/conftest.py
import pytest
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def test_data_path():
    """Path to test data directory."""
    return TEST_DATA_DIR

@pytest.fixture
def baltimore_test_data(test_data_path):
    """Load Baltimore test dataset."""
    conc_file = test_data_path / "baltimore_concentrations_small.csv"
    unc_file = test_data_path / "baltimore_uncertainties_small.csv"
    
    if not conc_file.exists():
        pytest.skip("Baltimore test data not available")
    
    return {
        'concentrations': pd.read_csv(conc_file, index_col=0),
        'uncertainties': pd.read_csv(unc_file, index_col=0)
    }
```

### Synthetic Data Generation

Create reproducible synthetic data:

```python
def generate_synthetic_pmf_data(
    n_samples: int = 100,
    n_species: int = 10, 
    n_factors: int = 3,
    noise_level: float = 0.1,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic PMF data for testing."""
    
    np.random.seed(random_seed)
    
    # Create factor profiles
    profiles = np.random.exponential(scale=1.0, size=(n_species, n_factors))
    
    # Create factor contributions
    contributions = np.random.exponential(scale=10.0, size=(n_samples, n_factors))
    
    # Generate concentrations
    concentrations = contributions @ profiles.T
    
    # Add noise
    noise = np.random.normal(0, noise_level * concentrations.mean(), 
                           size=concentrations.shape)
    concentrations += noise
    concentrations = np.maximum(concentrations, 0)  # Non-negative
    
    # Create uncertainties (10% + Poisson)
    uncertainties = np.sqrt(concentrations * 0.01**2 + concentrations)
    
    # Convert to DataFrames
    species_names = [f'Species_{i}' for i in range(n_species)]
    sample_names = [f'Sample_{i}' for i in range(n_samples)]
    
    conc_df = pd.DataFrame(concentrations, 
                          index=sample_names, 
                          columns=species_names)
    unc_df = pd.DataFrame(uncertainties,
                         index=sample_names,
                         columns=species_names)
    
    return conc_df, unc_df
```

## Debugging Tests

### Common Issues and Solutions

1. **Flaky Tests**
   ```python
   # Use fixed random seeds for reproducibility
   @pytest.fixture(autouse=True)
   def set_random_seed():
       np.random.seed(42)
       random.seed(42)
   ```

2. **Slow Tests**
   ```python
   # Use smaller datasets or mock expensive operations
   @pytest.fixture
   def small_dataset():
       return generate_synthetic_pmf_data(n_samples=20, n_species=5)
   ```

3. **Memory Issues**
   ```python
   # Clean up large objects
   def test_large_data_processing():
       large_data = create_large_dataset()
       try:
           result = process_data(large_data)
           assert result is not None
       finally:
           del large_data  # Explicit cleanup
   ```

### Test Debugging Tools

```bash
# Run single test with debugging
uv run pytest tests/unit/test_core.py::test_failing_function -vvv -s

# Drop into debugger on failure
uv run pytest --pdb

# Profile test execution time
uv run pytest --profile-svg
```

## Best Practices

### Test Writing Guidelines

1. **Test Names**: Use descriptive names that explain what is being tested
2. **Single Responsibility**: Each test should test one thing
3. **Independence**: Tests should not depend on each other
4. **Reproducibility**: Use fixed seeds and deterministic data
5. **Fast Execution**: Keep tests fast; use mocks for expensive operations

### Coverage Guidelines

- **Aim for >90% line coverage**
- **Test edge cases and error conditions**
- **Cover both happy path and failure scenarios**
- **Include integration tests for critical workflows**

### Maintenance

- **Regular cleanup**: Remove obsolete tests
- **Update fixtures**: Keep test data relevant
- **Monitor performance**: Track test execution time
- **Review coverage**: Ensure new code is tested

For more information on contributing, see the [Contributing Guidelines](guidelines.md).
