"""Does garch_fit reach the global maximum of its own likelihood?

Brute force on the identical objective: for each series, 78 Nelder-Mead
starts (scipy; a 6 x 6 x 3 grid less the 10 pairs at or over the persistence bound) on the Gaussian GARCH(1,1) negative log-likelihood evaluated
through the public filter with the fit's own backcast, against the single
call to garch_fit. The series are built to be awkward: near-IGARCH, no ARCH
effect, t(3) innovations, a 50-sigma outlier, a fourfold variance regime
switch, each at n = 1500 and n = 100, plus SPY from examples/data.

Needs scipy and the installed package; takes about a minute. Exit status is
non-zero if the fit is ever worse than brute force by more than 1e-8 relative.

    python tests/reference/garch_multistart_check.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import walkforward as wf
from scipy.optimize import minimize

MAX_PERSISTENCE = wf.GARCH_MAX_PERSISTENCE
ALPHAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
BETAS = (0.0, 0.3, 0.6, 0.8, 0.9, 0.97)
OMEGA_SCALES = (0.3, 1.0, 3.0)


def negative_loglik(theta: np.ndarray, r: np.ndarray, backcast: float) -> float:
    omega, alpha, beta = (float(v) for v in theta)
    if not (omega > 0.0 and alpha >= 0.0 and beta >= 0.0 and alpha + beta < MAX_PERSISTENCE):
        return math.inf
    sigma = wf.GarchModel(omega, alpha, beta, backcast=backcast).filter(r)
    sigma2 = sigma * sigma
    return 0.5 * float(np.sum(np.log(sigma2) + r * r / sigma2))


def brute_force(r: np.ndarray, backcast: float) -> tuple[float, np.ndarray]:
    best_f, best_x = math.inf, None
    for alpha in ALPHAS:
        for beta in BETAS:
            if alpha + beta >= MAX_PERSISTENCE:
                continue
            for scale in OMEGA_SCALES:
                x0 = np.array([scale * backcast * (1.0 - alpha - beta), alpha, beta])
                if x0[0] <= 0.0:
                    continue
                result = minimize(
                    negative_loglik,
                    x0,
                    args=(r, backcast),
                    method="Nelder-Mead",
                    options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000, "maxfev": 20000},
                )
                if result.fun < best_f:
                    best_f, best_x = float(result.fun), result.x
    return best_f, best_x


def simulate(kind: str, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "no ARCH effect":
        return rng.standard_normal(n) * 0.01
    if kind == "near-IGARCH":
        omega, alpha, beta = 1e-8, 0.06, 0.9395
    else:
        omega, alpha, beta = 2e-6, 0.10, 0.85
    z = rng.standard_t(3, n) / math.sqrt(3.0) if kind == "t(3) innovations" else rng.standard_normal(n)
    r = np.empty(n)
    sigma2 = omega / (1.0 - alpha - beta)
    for t in range(n):
        r[t] = math.sqrt(sigma2) * z[t]
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
    if kind == "50-sigma outlier":
        r[n // 2] = 50.0 * r.std()
    if kind == "variance regime switch":
        r[n // 2 :] *= 2.0
    return r


def series() -> list[tuple[str, np.ndarray]]:
    out = []
    for kind in ("moderate GARCH", "near-IGARCH", "no ARCH effect", "t(3) innovations",
                 "50-sigma outlier", "variance regime switch"):
        for n in (1500, 100):
            out.append((f"{kind}, n={n}", simulate(kind, n)))
    spy = pathlib.Path(__file__).resolve().parents[2] / "examples" / "data" / "SPY.csv"
    if spy.exists():
        close = np.loadtxt(spy, delimiter=",", skiprows=1, usecols=5)
        out.append(("SPY, last 2000 log returns", np.diff(np.log(close))[-2000:]))
    return out


def main() -> int:
    worst = 0.0
    print(f"{'series':34s} {'fit nll':>16s} {'brute nll':>16s} {'gap':>10s}  alpha, beta (fit)")
    for name, r in series():
        r = r - r.mean()
        model = wf.garch_fit(r)
        fit_f = -model.loglik
        brute_f, _ = brute_force(r, model.backcast)
        gap = fit_f - brute_f
        relative = gap / (abs(brute_f) + 1.0)
        worst = max(worst, relative)
        flag = "  WORSE" if relative > 1e-8 else ""
        print(f"{name:34s} {fit_f:16.8f} {brute_f:16.8f} {gap:+10.2e}  "
              f"{model.alpha:.4f}, {model.beta:.4f}{flag}")
    print(f"worst relative gap {worst:.1e}")
    return 0 if worst <= 1e-8 else 1


if __name__ == "__main__":
    sys.exit(main())
