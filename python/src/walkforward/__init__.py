"""Volatility forecasts, position sizing and purged walk-forward splits.

A thin, checked layer over the mlrisk C library. The organising idea is that
nothing at index ``t`` may see period ``t``:

* :func:`ewma_vol` and the GARCH filters are predictive. The value at ``t`` is
  built from observations before ``t``, so a position sized from it can be
  scored against ``returns[t]``.
* :func:`rolling_mean`, :func:`rolling_std`, :func:`parkinson_vol` and
  :func:`garman_klass_vol` are contemporaneous by construction, and say so.
  Pass them through :func:`lag` before sizing.
* :class:`PurgedWalkForward` keeps training labels out of the test window,
  which ordinary time-series splitting does not.

A worked loop::

    import numpy as np
    import walkforward as wf

    train = returns[:1000] - returns[:1000].mean()   # training mean only
    model = wf.garch_fit(train)

    sigma = model.filter_from(returns[1000:])        # continues the fit state
    position = wf.vol_target_position(
        sigma, target_vol=0.01, equity=100_000.0,
        price=close[999:-1], max_leverage=2.0,       # entry price is the prior close
    )
    pnl = position * close[999:-1] * returns[1000:]

This is research tooling, not investment advice, and not a backtester: it
knows nothing about costs, borrow, calendars or corporate actions.
"""

from __future__ import annotations

from ._core import DomainError, c_version
from .linear import Ridge
from .rolling import lag, rolling_mean, rolling_std
from .sizing import drawdown_scale, kelly_fraction, vol_target_position
from .split import PurgedWalkForward, WalkForwardSplit, walk_forward_splits
from .volatility import (
    GARCH_MAX_PERSISTENCE,
    GARCH_MIN_SAMPLE,
    GarchModel,
    ewma_vol,
    garch_fit,
    garman_klass_vol,
    parkinson_vol,
)

__all__ = [
    "DomainError",
    "GARCH_MAX_PERSISTENCE",
    "GARCH_MIN_SAMPLE",
    "GarchModel",
    "PurgedWalkForward",
    "Ridge",
    "WalkForwardSplit",
    "__version__",
    "c_version",
    "drawdown_scale",
    "ewma_vol",
    "garch_fit",
    "garman_klass_vol",
    "kelly_fraction",
    "lag",
    "parkinson_vol",
    "rolling_mean",
    "rolling_std",
    "vol_target_position",
    "walk_forward_splits",
]


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("walkforward")
    except Exception:  # pragma: no cover - source checkouts without metadata
        return c_version()


#: Version of the installed package. :func:`c_version` reports the version the
#: loaded shared library was built from; the two agree in a correct install.
__version__: str = _version()
