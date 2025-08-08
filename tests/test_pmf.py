"""Test suite for Easy PMF package.

Basic tests to ensure the PMF algorithm works correctly.
"""

import numpy as np
import pandas as pd
import pytest

from pmf_analysis import PMF


class TestPMF:
    """Test cases for the PMF class."""

    def setup_method(self):
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
        self.X = np.maximum(x_true + noise, 0.01)  # Ensure positive values

        # Create uncertainty matrix
        self.U = 0.1 * self.X + 0.01

        # Convert to DataFrames
        self.X_df = pd.DataFrame(
            self.X,
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=[f"species_{i}" for i in range(n_features)],
        )
        self.U_df = pd.DataFrame(
            self.U, index=self.X_df.index, columns=self.X_df.columns
        )

    def test_pmf_initialization(self):
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

    def test_pmf_fit_basic(self):
        """Test basic PMF fitting."""
        pmf = PMF(n_components=3, random_state=42)
        pmf.fit(self.X_df, self.U_df)

        # Check that results exist
        assert pmf.contributions_ is not None
        assert pmf.profiles_ is not None
        assert isinstance(pmf.contributions_, pd.DataFrame)
        assert isinstance(pmf.profiles_, pd.DataFrame)

        # Check shapes
        assert pmf.contributions_.shape == (self.X_df.shape[0], 3)
        assert pmf.profiles_.shape == (3, self.X_df.shape[1])

        # Check that all values are non-negative
        assert (pmf.contributions_.values >= 0).all()
        assert (pmf.profiles_.values >= 0).all()


def test_legacy_compatibility():
    """Test compatibility with the original test."""
    # Create a dummy dataset
    x = pd.DataFrame(np.random.rand(100, 10))

    # Create a PMF object
    pmf = PMF(n_components=5)

    # Fit the model
    pmf.fit(x)

    # Check that the factor matrices have the correct shape
    assert pmf.contributions_.shape == (100, 5)
    assert pmf.profiles_.shape == (5, 10)


def test_package_imports():
    """Test that the package can be imported correctly."""
    from pmf_analysis import PMF

    assert PMF is not None


if __name__ == "__main__":
    pytest.main([__file__])
