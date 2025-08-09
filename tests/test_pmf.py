"""Test suite for Easy PMF package.

Basic tests to ensure the PMF algorithm works correctly.
"""

import numpy as np
import pandas as pd
import pytest

from easy_pmf import PMF


class TestPMF:
    """Test cases for the PMF class."""

    def setup_method(self) -> None:
        """Set up test data."""
        # Create simple synthetic data
        np.random.seed(42)
        n_samples, n_features, n_components = 50, 10, 3

        # Generate synthetic factor matrices
        g_true = np.random.exponential(2, (n_samples, n_components))
        f_true = np.random.exponential(1, (n_components, n_features))

        # Generate synthetic data
        x_true = g_true @ f_true
        noise = np.random.normal(0, 0.1 * x_true.mean(), x_true.shape)
        x_array = np.maximum(x_true + noise, 0.01)  # Ensure positive values

        # Create uncertainty matrix
        u_array = 0.1 * x_array + 0.01

        # Convert to DataFrames
        self.X = pd.DataFrame(
            x_array,
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=[f"species_{i}" for i in range(n_features)],
        )
        self.U = pd.DataFrame(u_array, index=self.X.index, columns=self.X.columns)

        # Initialize PMF instance for tests
        self.pmf = PMF(n_components=3, random_state=42)

    def test_pmf_initialization(self) -> None:
        """Test PMF initialization with various parameters."""
        # Test basic initialization
        pmf = PMF(n_components=3)
        assert pmf.n_components == 3
        assert pmf.max_iter == 1000
        assert pmf.tol == 1e-4
        assert pmf.random_state is None

        # Test custom parameters
        pmf = PMF(n_components=5, max_iter=500, tol=1e-3, random_state=42)
        assert pmf.n_components == 5
        assert pmf.max_iter == 500
        assert pmf.tol == 1e-3
        assert pmf.random_state == 42

    def test_pmf_fit_basic(self) -> None:
        """Test basic PMF fitting functionality."""
        # Fit the model
        self.pmf.fit(self.X, self.U)

        # Check that the model was fitted
        assert self.pmf.contributions_ is not None
        assert self.pmf.profiles_ is not None

        # Check shapes - now we know they're not None
        contributions = self.pmf.contributions_
        profiles = self.pmf.profiles_
        assert contributions.shape == (self.X.shape[0], self.pmf.n_components)
        assert profiles.shape == (self.pmf.n_components, self.X.shape[1])


def test_legacy_compatibility() -> None:
    """Test compatibility with the original test."""
    # Create a dummy dataset
    x = pd.DataFrame(np.random.rand(100, 10))

    # Create a PMF object
    pmf = PMF(n_components=5)

    # Fit the model
    pmf.fit(x)

    # Check that the model was fitted
    assert pmf.contributions_ is not None
    assert pmf.profiles_ is not None

    # Check that the factor matrices have the correct shape
    assert pmf.contributions_.shape == (100, 5)
    assert pmf.profiles_.shape == (5, 10)


def test_package_imports() -> None:
    """Test that the package can be imported correctly."""
    from easy_pmf import PMF

    assert PMF is not None


if __name__ == "__main__":
    pytest.main([__file__])
