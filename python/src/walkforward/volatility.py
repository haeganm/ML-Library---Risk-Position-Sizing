"""Volatility forecasts.

Every forecast here is *predictive*: the value at index ``t`` is built from
observations strictly before ``t``, so a position sized from ``sigma[t]`` can
be scored against ``returns[t]`` without lookahead. The range estimators are
the exception and say so; they measure bar ``t`` from bar ``t`` itself and
must be lagged one period before they size anything.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._core import Garch, as_count, as_input, check, lib, like, out_like, ptr, same_length

__all__ = [
    "GARCH_MAX_PERSISTENCE",
    "GARCH_MIN_SAMPLE",
    "GarchModel",
    "ewma_vol",
    "garch_fit",
    "garman_klass_vol",
    "parkinson_vol",
]

#: Smallest sample :func:`garch_fit` accepts. Three parameters need a few
#: hundred observations to be identified; the floor only rules out the
#: hopeless cases.
GARCH_MIN_SAMPLE = 100

#: Feasibility bound on ``alpha + beta``. A model at or above this is
#: effectively integrated and its unconditional variance does not exist.
GARCH_MAX_PERSISTENCE = 0.9999


def ewma_vol(returns: Any, lam: float = 0.94) -> Any:
    """Exponentially weighted volatility forecast (RiskMetrics).

    ``out[t] = sqrt(sigma2[t])`` where
    ``sigma2[t] = lam * sigma2[t-1] + (1 - lam) * returns[t-1]**2``, so the
    forecast for period ``t`` uses returns strictly before ``t``.

    The first usable return seeds the variance, which makes the start of the
    series warmup: at ``lam = 0.94`` the seed still carries 5% of the weight
    fifty periods in. Missing observations are skipped rather than propagated,
    so a gap leaves the forecast it had rather than poisoning everything after
    it.

    Parameters
    ----------
    returns
        Period returns. A pandas Series keeps its index in the result.
    lam
        Decay factor in ``[0, 1)``. Daily work typically uses 0.94 (the
        RiskMetrics value, a half-life of about 11 days) through 0.97.

    Returns
    -------
    Per-period volatility, ``NaN`` until the first forecast exists.
    """
    array = as_input(returns, "returns")
    out = out_like(array.shape[0])
    check(
        lib.mlr_ewma_vol(ptr(array), array.shape[0], float(lam), ptr(out)),
        "ewma_vol",
        f"lam={lam!r} must be finite and in [0, 1)",
    )
    return like(out, returns)


@dataclass(frozen=True)
class GarchModel:
    """A fitted GARCH(1,1): ``sigma2[t] = omega + alpha*r[t-1]**2 + beta*sigma2[t-1]``.

    Returned by :func:`garch_fit`. Also constructible by hand from published
    parameters, in which case set ``sigma2_next`` before calling
    :meth:`forecast` and leave ``backcast`` at 0 to start the recursion from
    the unconditional variance.
    """

    omega: float
    alpha: float
    beta: float
    sigma2_next: float = 0.0
    loglik: float = float("nan")
    converged: bool = False
    backcast: float = 0.0

    # -- derived quantities a reader would otherwise compute by hand --------

    @property
    def persistence(self) -> float:
        """``alpha + beta``. How much of a variance shock survives one period."""
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """``omega / (1 - alpha - beta)``, the level the forecast decays to."""
        return self.omega / (1.0 - self.persistence)

    @property
    def unconditional_vol(self) -> float:
        """Square root of :attr:`unconditional_variance`, in return units."""
        return math.sqrt(self.unconditional_variance)

    @property
    def half_life(self) -> float:
        """Periods for a variance shock to decay halfway to the mean.

        ``log(0.5) / log(persistence)``. Infinite at unit persistence.
        """
        if self.persistence <= 0.0:
            return 0.0
        if self.persistence >= 1.0:
            return math.inf
        return math.log(0.5) / math.log(self.persistence)

    # -- the C calls -------------------------------------------------------

    def _struct(self) -> Garch:
        return Garch(
            omega=float(self.omega),
            alpha=float(self.alpha),
            beta=float(self.beta),
            sigma2_next=float(self.sigma2_next),
            loglik=float(self.loglik),
            converged=int(bool(self.converged)),
            backcast=float(self.backcast),
        )

    def _invalid(self) -> str:
        return (
            f"omega={self.omega!r} must be finite and positive, "
            f"alpha={self.alpha!r} and beta={self.beta!r} must be non-negative "
            f"with a sum below {GARCH_MAX_PERSISTENCE}"
        )

    def filter(self, returns: Any) -> Any:
        """Conditional volatility path, starting from this model's backcast.

        ``out[t]`` uses ``returns[:t]`` only, so filtering a prefix of a series
        returns exactly the prefix of the full filter.

        Use this for a series that begins where the fit sample began. To carry
        a fitted model onto the data that *follows* the fit sample, use
        :meth:`filter_from`, which continues from the variance state the fit
        ended on; filtering the new data alone restarts from the pre-sample
        seed and is wrong for the first few dozen periods.
        """
        array = as_input(returns, "returns")
        out = out_like(array.shape[0])
        model = self._struct()
        check(
            lib.mlr_garch_filter(ctypes.byref(model), ptr(array), array.shape[0], ptr(out)),
            "GarchModel.filter",
            self._invalid(),
        )
        return like(out, returns)

    def filter_from(self, returns: Any, sigma2_first: float | None = None) -> Any:
        """Continue the recursion from a known variance state.

        With the default ``sigma2_first`` the path picks up exactly where the
        fit sample ended, so the result equals the tail of filtering the fit
        sample and the new data together, bit for bit.

        Parameters
        ----------
        returns
            The returns that follow the fit sample.
        sigma2_first
            Conditional variance of the first period. Defaults to
            :attr:`sigma2_next`, which is the state at the end of the fit.
        """
        seed = self.sigma2_next if sigma2_first is None else float(sigma2_first)
        if not (seed > 0.0) or not math.isfinite(seed):
            raise ValueError(
                "sigma2_first must be finite and positive, got "
                f"{seed!r}. A hand-built model has sigma2_next = 0 until you set it."
            )
        array = as_input(returns, "returns")
        out = out_like(array.shape[0])
        model = self._struct()
        check(
            lib.mlr_garch_filter_from(
                ctypes.byref(model), seed, ptr(array), array.shape[0], ptr(out)
            ),
            "GarchModel.filter_from",
            self._invalid(),
        )
        return like(out, returns)

    def forecast(self, horizon: int) -> np.ndarray:
        """Volatility ``1..horizon`` periods past the end of the fit sample.

        ``out[h]`` is the forecast for period ``T + 1 + h``, decaying from
        :attr:`sigma2_next` toward :attr:`unconditional_vol` at rate
        :attr:`persistence`.
        """
        steps = as_count(horizon, "horizon", minimum=1)
        out = out_like(steps)
        model = self._struct()
        check(
            lib.mlr_garch_forecast(ctypes.byref(model), steps, ptr(out)),
            "GarchModel.forecast",
            f"sigma2_next={self.sigma2_next!r} must be finite and positive; " + self._invalid(),
        )
        return out


def garch_fit(returns: Any) -> GarchModel:
    """Fit GARCH(1,1) by Gaussian maximum likelihood.

    A coarse grid seeds Nelder-Mead from its three best points, each restarted
    until it stops improving, and the best result is kept; the likelihood can
    have more than one local maximum. Estimates do not depend on the units of
    the returns.

    The model assumes mean-zero returns. Subtract the mean of the *training
    window* before fitting, never the mean of the whole sample, which would
    put information from the future into the fit. A drift of 13% a year moves
    alpha by about 0.002.

    Parameters
    ----------
    returns
        At least :data:`GARCH_MIN_SAMPLE` finite, mean-zero returns.

    Raises
    ------
    ValueError
        The sample is too short or holds a non-finite value.
    DomainError
        The sample has zero, unrepresentable or denormally small variance.

    Notes
    -----
    ``converged`` reports that the optimizer met its tolerances, not that the
    model is identified. On a few hundred observations a GARCH fit converges
    happily to a near-unit-root model that means very little.
    """
    array = as_input(returns, "returns")
    if array.shape[0] < GARCH_MIN_SAMPLE:
        raise ValueError(
            f"garch_fit needs at least {GARCH_MIN_SAMPLE} returns to identify "
            f"three parameters, got {array.shape[0]}"
        )
    model = Garch()
    check(
        lib.mlr_garch_fit(ptr(array), array.shape[0], ctypes.byref(model)),
        "garch_fit",
        "every return must be finite and the sample must have representable variance",
    )
    return GarchModel(
        omega=model.omega,
        alpha=model.alpha,
        beta=model.beta,
        sigma2_next=model.sigma2_next,
        loglik=model.loglik,
        converged=bool(model.converged),
        backcast=model.backcast,
    )


def parkinson_vol(high: Any, low: Any) -> Any:
    """Parkinson range volatility, ``sqrt(ln(high/low)**2 / (4 ln 2))``.

    Contemporaneous: ``out[t]`` measures bar ``t``. Lag it one period before
    sizing a position held over bar ``t``.

    Bars that cannot produce an estimate (non-finite, non-positive, high below
    low, or a ratio that overflows) give ``NaN`` rather than failing the call.
    Range estimators are biased slightly low on discretely sampled bars, since
    the observed extremes understate the continuous ones.
    """
    h = as_input(high, "high")
    low_array = as_input(low, "low")
    same_length("high", h, "low", low_array)
    out = out_like(h.shape[0])
    check(lib.mlr_parkinson_vol(ptr(h), ptr(low_array), h.shape[0], ptr(out)), "parkinson_vol")
    return like(out, high)


def garman_klass_vol(open: Any, high: Any, low: Any, close: Any) -> Any:  # noqa: A002
    """Garman-Klass range volatility.

    ``sigma2 = 0.5 * ln(high/low)**2 - (2 ln 2 - 1) * ln(close/open)**2``,
    which uses the whole bar and is the more efficient of the two range
    estimators here.

    Contemporaneous: ``out[t]`` measures bar ``t``. Lag it one period before
    sizing a position held over bar ``t``.

    Bars whose open or close falls outside ``[low, high]`` are inconsistent
    and give ``NaN``, as do non-finite and non-positive prices. On a
    consistent bar the estimator cannot go negative.
    """
    o = as_input(open, "open")
    h = as_input(high, "high")
    low_array = as_input(low, "low")
    c = as_input(close, "close")
    for name, array in (("high", h), ("low", low_array), ("close", c)):
        same_length("open", o, name, array)
    out = out_like(o.shape[0])
    check(
        lib.mlr_garman_klass_vol(
            ptr(o), ptr(h), ptr(low_array), ptr(c), o.shape[0], ptr(out)
        ),
        "garman_klass_vol",
    )
    return like(out, open)
