"""Ridge regression.

The fit runs in C as a Householder QR on the centered design with
``sqrt(ridge) I`` appended, so accuracy is about the condition number times
epsilon rather than its square, and columns are shifted by their first row
before centering so a feature at a large level (a price near 1e9 with unit
variation) keeps its slope precision.

Prediction is a dot product, so it stays in numpy: the coefficients are
copied out and the C model is released as soon as the fit returns, which
means there is no native pointer to outlive its owner.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ._core import DomainError, LinModel, as_input, check, lib, ptr

try:  # sklearn is optional
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.exceptions import NotFittedError
except ImportError:  # pragma: no cover - exercised where sklearn is absent

    class _BaseEstimator:  # type: ignore[no-redef]
        pass

    class _RegressorMixin:  # type: ignore[no-redef]
        pass

    class NotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
        """Raised when predicting with a model that has not been fitted."""


try:  # pandas is optional; only used to remember DataFrame column names
    import pandas as _pd
except ImportError:  # pragma: no cover
    _pd = None


__all__ = ["Ridge"]


class Ridge(_RegressorMixin, _BaseEstimator):
    """Ridge regression with an unpenalised intercept.

    Minimises ``||Xc w - yc||^2 + ridge * ||w||^2`` on centered data, so the
    intercept is recovered afterwards rather than shrunk.

    Parameters
    ----------
    ridge
        Penalty on the squared coefficients. ``0`` is ordinary least squares
        and then the design must be full rank, which needs more rows than
        columns and linearly independent ones.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted weights, available after :meth:`fit`.
    intercept_ : float
        Fitted intercept.
    n_features_in_ : int
        Number of features seen at :meth:`fit`.
    feature_names_in_ : ndarray of str
        Column names, when ``X`` was a DataFrame with string columns.
        :meth:`predict` then requires the same columns in the same order,
        because a reordered frame would be silently misread.

    Raises
    ------
    DomainError
        The design is singular after centering (fewer rows than columns plus
        one, or a dependent column), or its arithmetic overflows. A
        rank-deficient design is refused rather than silently
        pseudo-inverted; add a ridge if that is what you want.

    Notes
    -----
    Intended for tens of features, not thousands: the solve is dense and
    unpivoted. Rank deficiency is detected on the diagonal of ``R``, which
    catches exact and near-exact dependence but is not a guarantee for every
    pathological design.
    """

    def __init__(self, ridge: float = 0.0) -> None:
        self.ridge = ridge

    def fit(self, X: Any, y: Any) -> Ridge:
        """Fit the model. Leaves the estimator untouched if the fit fails."""
        if isinstance(self.ridge, (str, bytes)) or not isinstance(self.ridge, (int, float, np.number)):
            raise TypeError(f"ridge must be a number, got {self.ridge!r}")
        ridge = float(self.ridge)
        if y is None:
            raise ValueError("Ridge.fit needs y; this is a supervised estimator")
        design = as_input(X, "X", ndim=2)
        target = as_input(y, "y")
        n, d = design.shape
        if target.shape[0] != n:
            raise ValueError(
                f"X has {n} rows but y has {target.shape[0]}; they must match"
            )
        if ridge == 0.0 and n <= d:
            raise DomainError(
                f"Ridge.fit: domain error (X is {n} by {d}: with ridge=0 the fit "
                "needs more rows than columns, because centering costs one rank; "
                "pass a positive ridge to fit anyway)"
            )

        model = LinModel()
        check(lib.mlr_lin_model_init(ctypes.byref(model), d), "Ridge.fit")
        try:
            check(
                lib.mlr_linreg_fit(ptr(design), ptr(target), n, d, ridge, ctypes.byref(model)),
                "Ridge.fit",
                f"ridge={ridge!r} must be finite and non-negative, and X and y must be finite",
                "the design is rank deficient after centering, or the solve overflowed; "
                "a dependent or constant column needs a positive ridge",
            )
            # Copy out and release: no native pointer survives this call
            self.coef_ = np.array([model.w[j] for j in range(d)], dtype=np.float64)
            self.intercept_ = float(model.b)
        finally:
            lib.mlr_lin_model_free(ctypes.byref(model))
        self.n_features_in_ = d
        names = _column_names(X)
        if names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = names
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict ``X @ coef_ + intercept_``."""
        if not hasattr(self, "coef_"):
            raise NotFittedError("this Ridge is not fitted yet; call fit first")
        names = _column_names(X)
        fitted_names = getattr(self, "feature_names_in_", None)
        if names is not None and fitted_names is not None and not np.array_equal(names, fitted_names):
            raise ValueError(
                "X has columns "
                f"{list(names)} but the model was fitted on {list(fitted_names)}; "
                "the same columns in the same order are required"
            )
        design = as_input(X, "X", ndim=2)
        if design.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {design.shape[1]} features but the model was fitted on "
                f"{self.coef_.shape[0]}"
            )
        return design @ self.coef_ + self.intercept_


def _column_names(X: Any) -> np.ndarray | None:
    """String column names of a DataFrame, else None."""
    if _pd is None or not isinstance(X, _pd.DataFrame):
        return None
    columns = list(X.columns)
    if not all(isinstance(c, str) for c in columns):
        return None
    return np.asarray(columns, dtype=object)
