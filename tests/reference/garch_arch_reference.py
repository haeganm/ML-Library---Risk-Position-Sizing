"""Reference GARCH(1,1) fits from the Python ``arch`` package.

Regenerates the constants used by ``test_garch_fit_matches_arch`` in
``tests/test_vol.c``. The simulated series is produced here with the same
64-bit LCG + Box-Muller generator as ``tests/test_util.h``, so the two sides
fit the same sample up to libm rounding of ``log`` and ``cos`` (a few ulps,
absorbed by the C test tolerances); ``arch`` is given the identical variance
backcast (mean of squared returns) so the two log-likelihoods are the same
function.

Returns are passed to ``arch`` scaled by 100: on raw daily-sized returns its
SLSQP optimizer stalls at the starting values (a documented limitation, hence
its rescale warning), while mlrisk fits the raw series directly.

Usage:
    pip install arch numpy
    python tests/reference/garch_arch_reference.py
"""

import math

import numpy as np
from arch.univariate import GARCH, Normal, ZeroMean

MASK = (1 << 64) - 1


def lcg_u01(state):
    state = (state * 6364136223846793005 + 1442695040888963407) & MASK
    return state, ((state >> 11) + 0.5) / 9007199254740992.0


def gauss(state):
    state, u1 = lcg_u01(state)
    state, u2 = lcg_u01(state)
    return state, math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * 3.14159265358979323846 * u2)


def simulate(seed, omega, alpha, beta, n):
    state = seed
    s2 = omega / (1.0 - alpha - beta)
    out = []
    for _ in range(n):
        state, z = gauss(state)
        r = math.sqrt(s2) * z
        out.append(r)
        s2 = omega + alpha * r * r + beta * s2
    return np.array(out)


def loglik(r, omega, alpha, beta, backcast):
    """mlrisk's objective: Gaussian log-likelihood with constants dropped."""
    s2 = omega + (alpha + beta) * backcast
    acc = 0.0
    for x in r:
        acc += math.log(s2) + x * x / s2
        s2 = omega + alpha * x * x + beta * s2
    return -0.5 * acc


CASES = [
    # (name, seed, omega, alpha, beta, n)
    ("moderate persistence", 12345, 2e-6, 0.10, 0.85, 2000),
    ("high persistence", 777, 5e-6, 0.05, 0.93, 1500),
]

SCALE = 100.0

if __name__ == "__main__":
    for name, seed, omega, alpha, beta, n in CASES:
        r = simulate(seed, omega, alpha, beta, n)
        backcast = float(np.mean(r * r))
        model = ZeroMean(r * SCALE, rescale=False)
        model.volatility = GARCH(p=1, q=1)  # positional GARCH(1, 1) would set o=1 (GJR)
        model.distribution = Normal()
        res = model.fit(
            disp="off",
            backcast=backcast * SCALE**2,
            options={"ftol": 1e-15, "maxiter": 5000},
            tol=1e-13,
        )
        assert list(res.params.index) == ["omega", "alpha[1]", "beta[1]"]
        om = res.params["omega"] / SCALE**2
        al = res.params["alpha[1]"]
        be = res.params["beta[1]"]
        ll = loglik(r, om, al, be, backcast)
        arch_ll = res.loglikelihood + 0.5 * n * math.log(2 * math.pi) + n * math.log(SCALE)
        assert abs(ll - arch_ll) < 1e-6, (ll, arch_ll)
        print(f"// {name}, arch {__import__('arch').__version__}, convergence_flag={res.convergence_flag}")
        print(f"{{{seed}, {omega:g}, {alpha:g}, {beta:g}, {n},")
        print(f" {om:.12e}, {al:.12f}, {be:.12f}, {ll:.6f}}},")
