"""Easy PMF - An easy-to-use package for Positive Matrix Factorization (PMF) analysis.

This package provides a simple interface for performing PMF analysis on
environmental data, with built-in visualization capabilities and support for
multiple dataset formats.
"""

__version__ = "0.1.0"
__author__ = "PMF Analysis Team"
__email__ = "contact@easy-pmf.org"

import warnings
from typing import Optional

import numpy as np
import pandas as pd


class PMF:
    """Positive Matrix Factorization (PMF).

    This class implements the PMF algorithm, which is used to decompose a matrix
    into two non-negative matrices for source apportionment analysis.

    Parameters
    ----------
    n_components : int
        The number of components (factors) to extract.
    max_iter : int, default=1000
        The maximum number of iterations to perform.
    tol : float, default=1e-4
        The tolerance for the stopping condition.
    random_state : int, default=None
        The seed for the random number generator for reproducible results.

    Attributes:
    ----------
    contributions_ : pandas.DataFrame
        The factor contributions (G matrix) - how much each factor contributes
        to each sample over time.
    profiles_ : pandas.DataFrame
        The factor profiles (F matrix) - the chemical signature of each factor.
    n_iter_ : int
        The number of iterations performed during fitting.
    converged_ : bool
        Whether the algorithm converged within the specified tolerance.

    Examples:
    --------
    >>> import pandas as pd
    >>> from pmf_analysis import PMF
    >>>
    >>> # Load your concentration and uncertainty data
    >>> concentrations = pd.read_csv("concentrations.csv", index_col=0)
    >>> uncertainties = pd.read_csv("uncertainties.csv", index_col=0)
    >>>
    >>> # Initialize PMF with 5 factors
    >>> pmf = PMF(n_components=5, random_state=42)
    >>>
    >>> # Fit the model
    >>> pmf.fit(concentrations, uncertainties)
    >>>
    >>> # Access results
    >>> factor_contributions = pmf.contributions_
    >>> factor_profiles = pmf.profiles_
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """Initialize the PMF model.

        Parameters are described in the class docstring.
        """
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if tol <= 0:
            raise ValueError("tol must be a positive number")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Initialize attributes that will be set during fitting
        self.contributions_ = None
        self.profiles_ = None
        self.n_iter_ = None
        self.converged_ = False

    def fit(self, x: pd.DataFrame, u: Optional[pd.DataFrame] = None) -> "PMF":
        """Fit the PMF model to the data.

        Parameters
        ----------
        x : pandas.DataFrame
            The concentration data to fit the model to. Rows are samples (time points),
            columns are chemical species.
        u : pandas.DataFrame, optional
            The uncertainty data corresponding to x. Must have the same shape as x.
            If not provided, uniform uncertainty of 1.0 will be used for all
            data points.

        Returns:
        -------
        self : PMF
            Returns the fitted PMF instance.

        Raises:
        ------
        ValueError
            If x contains negative values, if x and u have different shapes,
            or if x contains NaN values.
        """
        # Input validation
        if not isinstance(x, pd.DataFrame):
            raise TypeError("x must be a pandas DataFrame")

        if x.isnull().any().any():
            raise ValueError(
                "x contains NaN values. Please handle missing data before fitting."
            )

        if (x < 0).any().any():
            raise ValueError(
                "x contains negative values. PMF requires non-negative data."
            )

        if u is not None:
            if not isinstance(u, pd.DataFrame):
                raise TypeError("u must be a pandas DataFrame")
            if x.shape != u.shape:
                raise ValueError("x and u must have the same shape")
            if (u <= 0).any().any():
                warnings.warn(
                    "u contains zero or negative values. These will be replaced "
                    "with a small positive value.",
                    stacklevel=2,
                )

        # Store original index and columns for later use
        self._X_index = x.index
        self._X_columns = x.columns

        # Convert the data to numpy arrays
        x_array = x.values.astype(float)
        if u is not None:
            u_array = u.values.astype(float)
        else:
            u_array = np.ones_like(x_array)

        # Replace zeros or small values in u with a small number to
        # avoid division by zero
        u_array[u_array <= 0] = 1e-9

        # Get the dimensions of the data
        n_samples, n_features = x_array.shape

        # Initialize the random number generator
        rng = np.random.default_rng(self.random_state)

        # Initialize the factor matrices with small positive random values
        g = rng.random((n_samples, self.n_components)) + 1e-6
        f = rng.random((self.n_components, n_features)) + 1e-6

        # Pre-calculate the squared inverse of u
        u_inv_sq = 1 / (u_array**2)

        # Store convergence history for debugging
        self._convergence_history = []

        # Iterate until convergence
        for _i in range(self.max_iter):
            # Store old g and f for convergence check
            g_old = g.copy()
            f_old = f.copy()

            # Update the f matrix (profiles)
            numerator = g.T @ (x_array * u_inv_sq)
            denominator = g.T @ ((g @ f) * u_inv_sq)
            # Avoid division by zero
            denominator[denominator == 0] = 1e-10
            f = f * numerator / denominator

            # Update the g matrix (contributions)
            numerator = (x_array * u_inv_sq) @ f.T
            denominator = ((g @ f) * u_inv_sq) @ f.T
            # Avoid division by zero
            denominator[denominator == 0] = 1e-10
            g = g * numerator / denominator

            # Check for convergence
            g_diff = np.linalg.norm(g - g_old) / (np.linalg.norm(g_old) + 1e-10)
            f_diff = np.linalg.norm(f - f_old) / (np.linalg.norm(f_old) + 1e-10)

            total_diff = max(g_diff, f_diff)
            self._convergence_history.append(total_diff)

            if total_diff < self.tol:
                self.converged_ = True
                break

        # Store the number of iterations performed
        self.n_iter_ = _i + 1

        # Save the factor matrices as DataFrames with proper indices and columns
        self.contributions_ = pd.DataFrame(
            g,
            index=self._X_index,
            columns=[f"Factor_{i + 1}" for i in range(self.n_components)],
        )

        self.profiles_ = pd.DataFrame(
            f,
            index=[f"Factor_{i + 1}" for i in range(self.n_components)],
            columns=self._X_columns,
        )

        if not self.converged_:
            warnings.warn(
                f"PMF did not converge after {self.max_iter} iterations. "
                f"Consider increasing max_iter or adjusting tolerance.",
                stacklevel=2,
            )

        return self

    def transform(
        self, x: pd.DataFrame, u: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Transform new data using the fitted PMF model.

        Parameters
        ----------
        x : pandas.DataFrame
            New concentration data to transform.
        U : pandas.DataFrame, optional
            Uncertainty data for X.

        Returns:
        -------
        contributions : pandas.DataFrame
            Factor contributions for the new data.
        """
        if self.profiles_ is None:
            raise ValueError("Model must be fitted before transforming data")

        # This is a simplified transformation - in practice, you might want
        # to implement a more sophisticated approach
        if not x.columns.equals(self._X_columns):
            raise ValueError("x must have the same columns as the training data")

        # Use least squares to find contributions given fixed profiles
        f = self.profiles_.values
        x_array = x.values

        # Solve for g: x ≈ g @ f
        g, _, _, _ = np.linalg.lstsq(f.T, x_array.T, rcond=None)
        g = np.maximum(g.T, 0)  # Ensure non-negativity

        return pd.DataFrame(
            g,
            index=x.index,
            columns=[f"Factor_{i + 1}" for i in range(self.n_components)],
        )

    def score(self, x: pd.DataFrame, u: Optional[pd.DataFrame] = None) -> float:
        """Calculate the goodness of fit (Q value) for the PMF model.

        Parameters
        ----------
        x : pandas.DataFrame
            Concentration data.
        u : pandas.DataFrame, optional
            Uncertainty data.

        Returns:
        -------
        q_value : float
            The Q value (lower is better).
        """
        if self.contributions_ is None or self.profiles_ is None:
            raise ValueError("Model must be fitted before scoring")

        x_array = x.values
        if u is not None:
            u_array = u.values
        else:
            u_array = np.ones_like(x_array)

        # Replace zeros with small values
        u_array[u_array <= 0] = 1e-9

        # Calculate reconstructed data
        x_reconstructed = self.contributions_.values @ self.profiles_.values

        # Calculate Q value (weighted sum of squared residuals)
        residuals = (x_array - x_reconstructed) / u_array
        q_value = np.sum(residuals**2)

        return q_value


# Make the main class available at package level
__all__ = ["PMF"]
