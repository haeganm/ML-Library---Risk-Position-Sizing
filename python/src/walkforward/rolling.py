"""Rolling window statistics.

Trailing windows: ``out[i]`` is computed from ``x[i-window+1 : i+1]``, which
*includes* ``x[i]``. A rolling statistic at index ``t`` therefore knows period
``t`` and must be lagged one period before it sizes a position held over
``t``; :func:`lag` does that. Indices without a full window are ``NaN``, and
so is any window holding a non-finite value or whose arithmetic overflows.

Accuracy does not degrade at index levels. The accumulators are shifted by an
offset taken from inside the current window and rebuilt periodically, so a
price series near 1e9, a long trend, and a bad tick that has since left the
window all keep full precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import as_count, as_input, check, lib, like, out_like, ptr

__all__ = ["lag", "rolling_mean", "rolling_std"]


def _rolling(function: Any, name: str, x: Any, window: int) -> Any:
    array = as_input(x, "x")
    size = as_count(window, "window", minimum=1)
    out = out_like(array.shape[0])
    check(function(ptr(array), array.shape[0], size, ptr(out)), name)
    return like(out, x)


def rolling_mean(x: Any, window: int) -> Any:
    """Trailing mean over ``window`` observations, O(n).

    Contemporaneous: ``out[t]`` includes ``x[t]``. See :func:`lag`.
    """
    return _rolling(lib.mlr_rolling_mean, "rolling_mean", x, window)


def rolling_std(x: Any, window: int) -> Any:
    """Trailing standard deviation over ``window`` observations, O(n).

    Population convention, dividing by ``window``. Note that pandas
    ``rolling().std()`` defaults to the sample convention and divides by
    ``window - 1``, so the two differ by ``sqrt(window / (window - 1))``.

    Contemporaneous: ``out[t]`` includes ``x[t]``. See :func:`lag`.
    """
    return _rolling(lib.mlr_rolling_std, "rolling_std", x, window)


def lag(x: Any, periods: int = 1) -> Any:
    """Shift a series forward, filling the start with ``NaN``.

    The blessed way to turn a contemporaneous estimator into something safe to
    size with: ``lag(rolling_std(returns, 20))`` at index ``t`` holds the value
    computed through ``t-1``, so it is known when the position for ``t`` is
    entered.

    Parameters
    ----------
    x
        The series to shift. A pandas Series keeps its index.
    periods
        How many periods to shift forward. Must be non-negative.
    """
    array = as_input(x, "x")
    shift = as_count(periods, "periods", minimum=0)
    out = np.full(array.shape[0], np.nan, dtype=np.float64)
    if shift < array.shape[0]:
        out[shift:] = array[: array.shape[0] - shift]
    return like(out, x)
