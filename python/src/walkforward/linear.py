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

from ._core import LinModel, as_input, check, lib, ptr

try:  # sklearn is optional
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import RegressorMixin as _RegressorMixin
except ImportError:  # pragma: no cover - exercised where sklearn is absent

    class _BaseEstimator:  # type: ignore[no-redef]
        pass

    class _RegressorMixin:  # type: ignore[no-redef]
        pass


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

    Raises
    ------
    DomainError
        The design is singular, or its arithmetic overflows. A rank-deficient
        design is refused rather than silently pseudo-inverted; add a ridge if
        that is what you want.

    Notes
    -----
    Intended for tens of features, not thousands: the solve is dense and
    unpivoted. Rank deficiency is detected on the diagonal of ``R``, which
    catches exact and near-exact dependence but is not a guarantee for every
    pathological design.
    """

    def __init__(self, ridge: float = 0.0) -> None:
        self.ridge = ridge

    def fit(self, X: Any, y: Any) -> "Ridge":
        """Fit the model. Leaves the estimator untouched if the fit fails."""
        design = as_input(X, "X", ndim=2)
        target = as_input(y, "y")
        n, d = design.shape
        if target.shape[0] != n:
            raise ValueError(
                f"X has {n} rows but y has {target.shape[0]}; they must match"
            )
        if self.ridge == 0.0 and n <= d:
            raise ValueError(
                f"X is {n} by {d}: with ridge=0 the fit needs more rows than "
                "columns, because centering costs one rank. Pass a positive "
                "ridge to fit anyway."
            )

        model = LinModel()
        check(lib.mlr_lin_model_init(ctypes.byref(model), d), "Ridge.fit")
        try:
            check(
                lib.mlr_linreg_fit(
                    ptr(design), ptr(target), n, d, float(self.ridge), ctypes.byref(model)
                ),
                "Ridge.fit",
                f"ridge={self.ridge!r} must be finite and non-negative, "
                "and X and y must be finite",
            )
            # Copy out and release: no native pointer survives this call
            self.coef_ = np.array([model.w[j] for j in range(d)], dtype=np.float64)
            self.intercept_ = float(model.b)
        finally:
            lib.mlr_lin_model_free(ctypes.byref(model))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict ``X @ coef_ + intercept_``."""
        if not hasattr(self, "coef_"):
            raise ValueError("this Ridge is not fitted yet; call fit first")
        design = as_input(X, "X", ndim=2)
        if design.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {design.shape[1]} features but the model was fitted on "
                f"{self.coef_.shape[0]}"
            )
        return design @ self.coef_ + self.intercept_
