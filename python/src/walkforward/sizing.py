"""Turning a volatility forecast into a position.

Timing contract, which the whole package is built around:

* ``sigma[t]`` must be a forecast for period ``t`` made from information
  available at the close of ``t-1``. :func:`~walkforward.ewma_vol` and the
  GARCH filters already are; rolling statistics and range estimators are not
  and must be lagged first.
* ``position[t]`` is then the position, in units, entered at the close of
  ``t-1`` at ``price[t-1]`` and held over period ``t``.
* Its profit is ``position[t] * price[t-1] * returns[t]``. The price factor is
  easy to drop and doing so silently rescales every result.

So the price passed for index ``t`` is the *previous* close:
``vol_target_position(sigma[1:], ..., price=close[:-1], ...)``, or equivalently
``lag(close)`` aligned to the same index.
"""

from __future__ import annotations

import ctypes
from typing import Any

from ._core import as_input, check, lib, like, out_like, ptr, same_index, same_length

__all__ = ["drawdown_scale", "kelly_fraction", "vol_target_position"]


def vol_target_position(
    sigma: Any,
    target_vol: float,
    equity: float,
    price: Any,
    max_leverage: float,
) -> Any:
    """Size a position so its volatility matches a target.

    ``position = (target_vol / sigma) * (equity / price)``, capped so that
    ``position * price <= max_leverage * equity``.

    Read ``target_vol`` as a risk cap instead and the same formula caps risk
    per position: both are "the fraction of equity you are willing to see move
    in one period".

    Parameters
    ----------
    sigma
        Per-period volatility forecast for each period. Must be known before
        the period starts; see the module docstring.
    target_vol
        Target per-period volatility as a fraction of equity. Annualised
        targets convert as ``annual / sqrt(periods_per_year)``: 10% a year on
        daily data is ``0.10 / sqrt(252)``, about 0.0063.
    equity
        Account equity. Finite and positive.
    price
        Entry price for each period, that is the previous close.
    max_leverage
        Cap on notional over equity. ``max_leverage * equity`` must be finite.

    Returns
    -------
    Positions in units. A period whose sigma or price is non-finite,
    non-positive, or extreme enough to make the position unrepresentable gets
    a position of zero rather than a bad number.

    Two pandas Series must share an index. They are read by position, so a
    sliced ``close[:-1]`` against an unsliced ``sigma`` would be paired one
    bar off and the result stamped with ``sigma``'s index; use
    :func:`~walkforward.lag`, which keeps the index.
    """
    same_index("sigma", sigma, "price", price)
    sigma_array = as_input(sigma, "sigma")
    price_array = as_input(price, "price")
    same_length("sigma", sigma_array, "price", price_array)
    out = out_like(sigma_array.shape[0])
    check(
        lib.mlr_vol_target_position(
            ptr(sigma_array),
            float(target_vol),
            float(equity),
            ptr(price_array),
            float(max_leverage),
            sigma_array.shape[0],
            ptr(out),
        ),
        "vol_target_position",
        f"target_vol={target_vol!r}, equity={equity!r} and max_leverage={max_leverage!r} "
        "must be finite and positive, and max_leverage * equity must be finite",
    )
    return like(out, sigma)


def kelly_fraction(returns: Any, fraction: float = 1.0) -> float:
    """Mean-variance Kelly fraction of equity.

    ``fraction * mean(returns) / var(returns)`` with the sample variance
    (``n - 1`` denominator). Negative when the sample edge is negative, which
    the caller should read as "no position".

    This is a sizing utility, not an allocation model. It uses the raw
    historical mean rather than the excess over a funding rate, and it knows
    nothing about estimation error, fat tails, drawdown tolerance or the rest
    of the book. Mean over variance from a short sample is a strongly
    upward-biased estimate of the true edge, so treat the result as an upper
    bound and pass a ``fraction`` well below 1. Half Kelly (0.5) gives up a
    quarter of the growth rate for half the volatility, which is why it is the
    usual starting point.

    Parameters
    ----------
    returns
        At least two finite per-period returns.
    fraction
        Fractional Kelly multiplier: 1.0 is full Kelly, 0.5 is half.

    Raises
    ------
    ValueError
        Fewer than two returns, a non-finite return, or a ``fraction`` that
        is not finite and positive.
    DomainError
        The sample has zero variance, or its variance or the estimate cannot
        be represented.
    """
    array = as_input(returns, "returns")
    if array.shape[0] < 2:
        raise ValueError(f"kelly_fraction needs at least 2 returns, got {array.shape[0]}")
    out = ctypes.c_double()
    check(
        lib.mlr_kelly_fraction(
            ptr(array), array.shape[0], float(fraction), ctypes.byref(out)
        ),
        "kelly_fraction",
        f"fraction={fraction!r} must be finite and positive, and every return finite",
        "the sample has zero variance, or its mean, variance or the estimate "
        "cannot be represented",
    )
    return out.value


def drawdown_scale(equity: Any, max_dd: float) -> Any:
    """Exposure multiplier that tapers to zero as drawdown deepens.

    With running peak ``P[i] = max(equity[:i+1])`` and drawdown
    ``dd[i] = 1 - equity[i] / P[i]``, the result is
    ``clip(1 - dd / max_dd, 0, 1)``: full exposure at a new high, nothing at
    ``max_dd``.

    Contemporaneous, and easy to misuse because it looks like a multiplier you
    apply in place. ``scale[t]`` comes from ``equity[t]``, the close at the
    *end* of period ``t``, so it is not known when the position for ``t`` is
    entered. Scale ``position[t]`` by ``scale[t-1]``: use
    :func:`~walkforward.lag`. Applying ``scale[t]`` to ``position[t]``
    de-levers on the bar of a loss using that bar's own close, which flatters
    a backtest.

    Parameters
    ----------
    equity
        Cumulative equity path, every value finite and positive.
    max_dd
        Drawdown at which exposure reaches zero, in ``(0, 1]``.

    Raises
    ------
    ValueError
        A non-finite equity value, or ``max_dd`` outside ``(0, 1]``.
    DomainError
        An equity value that is zero or negative: there is no drawdown from
        a peak of nothing.
    """
    array = as_input(equity, "equity")
    out = out_like(array.shape[0])
    check(
        lib.mlr_drawdown_scale(ptr(array), array.shape[0], float(max_dd), ptr(out)),
        "drawdown_scale",
        f"max_dd={max_dd!r} must be finite and in (0, 1], and every equity value finite",
        "every equity value must be positive",
    )
    return like(out, equity)
