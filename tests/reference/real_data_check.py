#!/usr/bin/env python3
"""Run mlrisk over real daily OHLC files and report what it does with them.

Usage: python tests/reference/real_data_check.py FILE.csv [FILE.csv ...]

Each CSV needs Date, Open, High, Low, Close columns (yfinance / stooq style).
Reports, per series: bar hygiene (how many bars have open or close outside
[low, high]), GARCH fit vs arch on the last 2000 returns, scale invariance
of the fit on raw vs x100 returns, a bitwise prefix check of the filters,
range-estimator NaN counts, and the realized-vol/target ratio of the
walk-forward sizing loop. Not part of CI (needs data on disk); the numbers
quoted in the README came from this script.
"""

from __future__ import annotations

import os
import sys
import math
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reference_check import (L, Garch, garch_fit, garch_filter, ewma, rolling_std, vol_target,  # noqa: E402
                             splits, arr, ptr, OK, max_abs)


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.set_index(pd.to_datetime(df[date_col])).sort_index()
    return df[["open", "high", "low", "close"]].astype(float).dropna()


def fit_vs_arch(r):
    from arch.univariate import GARCH, Normal, ZeroMean
    ours = garch_fit(r)
    am = ZeroMean(r * 100, rescale=False); am.volatility = GARCH(p=1, q=1); am.distribution = Normal()
    res = am.fit(disp="off", backcast=float(np.mean(r * r)) * 1e4, options={"ftol": 1e-15, "maxiter": 5000}, tol=1e-13)
    ll_arch = res.loglikelihood + 0.5 * len(r) * math.log(2 * math.pi) + len(r) * math.log(100)
    return ours, res.params["alpha[1]"], res.params["beta[1]"], ll_arch


def main(paths):
    print("%-12s %6s %7s %8s %8s %8s %9s %9s %10s %7s %8s" % (
        "series", "bars", "badbars", "alpha", "beta", "a+b", "|da|arch", "ll-llarch", "scale a+b", "GK NaN", "vol/tgt"))
    for path in paths:
        name = pathlib.Path(path).stem
        df = load(path)
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        bad = int(np.sum((o < l) | (o > h) | (c < l) | (c > h) | (h < l)))
        r = np.diff(np.log(c))
        r = r[np.isfinite(r)]
        tail = r[-2000:] - r[-2000:].mean()

        ours, a_arch, b_arch, ll_arch = fit_vs_arch(tail)
        scaled = garch_fit(tail * 100)
        scale_diff = abs((scaled.alpha + scaled.beta) - (ours.alpha + ours.beta))

        # prefix stability on the real series, bitwise
        half = len(r) // 2
        m = Garch(omega=ours.omega, alpha=ours.alpha, beta=ours.beta, backcast=ours.backcast, converged=1)
        assert np.array_equal(garch_filter(m, r)[:half], garch_filter(m, r[:half]))
        e_full, e_half = ewma(r, 0.94), ewma(r[:half], 0.94)
        assert np.array_equal(np.nan_to_num(e_full[:half]), np.nan_to_num(e_half))

        out = np.empty(len(c))
        assert L.mlr_garman_klass_vol(ptr(arr(o)), ptr(arr(h)), ptr(arr(l)), ptr(arr(c)), len(c), ptr(out)) == OK
        gk_nan = int(np.isnan(out).sum())

        # rolling std at the real price level vs exact arithmetic (spot check)
        from fractions import Fraction
        w = 20; i = len(c) - 1
        win = [Fraction(v) for v in c[i - w + 1:i + 1]]; mu = sum(win) / w
        exact = math.sqrt(sum((v - mu) ** 2 for v in win) / w)
        assert abs(rolling_std(c, w)[-1] - exact) < 1e-9 * exact

        # walk-forward vol targeting on the real returns
        ratios = []
        _, sp = splits(len(r), 1000, 250, 250, 0, 0, 0)
        for (a, b, cc, d, _, _) in sp[-8:]:
            mm = garch_fit(r[a:b] - r[a:b].mean())
            sig = garch_filter(mm, r[a:d] - r[a:b].mean())
            pos = vol_target(sig[cc - a:d - a], 0.01, 1e5, c[cc - 1:d - 1], 3.0)
            gain = pos * c[cc - 1:d - 1] * (np.exp(r[cc:d]) - 1)
            ratios.append(np.std(gain / 1e5) / 0.01)

        print("%-12s %6d %7d %8.4f %8.4f %8.4f %9.1e %+9.1e %10.1e %7d %8.2f" % (
            name, len(c), bad, ours.alpha, ours.beta, ours.alpha + ours.beta, abs(ours.alpha - a_arch),
            ours.loglik - ll_arch, scale_diff, gk_nan, float(np.mean(ratios))))


if __name__ == "__main__":
    main(sys.argv[1:])
