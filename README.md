# mlrisk

> Volatility forecasts that have not seen the period they forecast, position sizes built from them, and walk-forward splits whose training labels cannot reach into the test window. C11, zero dependencies beyond libm.

[![CI](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml/badge.svg)](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml)
[![C11](https://img.shields.io/badge/C-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- **One timing convention.** Every forecast at index `t` is built from returns before `t`. EWMA and the GARCH filter both work this way, so a position sized from `sigma[t]` can be scored against `returns[t]` without a one-bar lookahead. The library tests this by construction: recompute on a prefix, get the prefix, bit for bit.
- **GARCH(1,1) by maximum likelihood** with no optimizer dependency, checked against the Python `arch` package on identical samples and identical likelihoods, and invariant to the units of the returns (a fit on returns scaled by 1e-8 gives the same alpha and beta as one scaled by 1e6).
- **Sizing that fails closed.** Volatility targeting with a notional cap, mean-variance Kelly, drawdown-based exposure scaling. A NaN price or sigma gives a zero position; a NaN equity or leverage refuses the call. Nothing turns into a NaN position with `MLR_OK`.
- **Purged, embargoed walk-forward splits** with a count-query API and no `size_t` arithmetic that can wrap.
- **Rolling mean and standard deviation** in O(n) that stay accurate at index levels (1.7e-15 at a price level of 1e9, where a plain two-pass computation is already off by 4.9e-12), along a trend from 100 to 1e6, and after a bad tick has left the window.
- **Ridge regression** for small feature sets by Householder QR on the centered design, so the intercept is unpenalized and accuracy stays at condition number times epsilon where normal equations would square it.

Builds as strict ISO C11 under GCC, Clang and MSVC with `-Wall -Wextra -Wpedantic -Werror` and `-ffp-contract=off` (`/W4 /WX /fp:precise /fp:contract-` on MSVC), so results agree across compilers configured for no fused multiply-add to the last bit. The public headers need only C99. CI runs the test suite on Linux (gcc and clang, 64- and 32-bit), macOS and Windows, under AddressSanitizer and UBSan, installs the library and consumes it through `find_package` and `pkg-config`, compiles the public headers as C++17, and runs a Python job that compares every function against pandas, numpy, scikit-learn and `arch`. This is research software, not investment advice.

## Build

Requires CMake 3.21 and any C11 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # strict C11, warnings are errors
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure   # six modules, one ctest entry each
./build/vol_target_demo                                  # walk-forward GARCH sizing loop, seed 42
./build/vol_target_demo 7                                # any other seed
```

On Windows with the Visual Studio generator the binaries land in `build\Release\`.

Consume it from CMake as a subdirectory or after `cmake --install build`:

```cmake
add_subdirectory(path/to/mlrisk)          # or find_package(mlrisk 3 REQUIRED) after install
target_link_libraries(app PRIVATE mlrisk::mlrisk)
```

`mlrisk.pc` is installed for pkg-config. When mlrisk is not the top-level project the tests, example, install rules and `-Werror` are all off by default (`MLRISK_BUILD_TESTS`, `MLRISK_BUILD_EXAMPLES`, `MLRISK_INSTALL`, `MLRISK_WERROR`).

## Use

The whole library is in service of this loop. `returns[t]` is the return over period `t` and `prices[t]` the close of period `t`.

```c
#include "mlrisk/mlrisk.h"

enum { N = 1500, TRAIN = 500 };
double returns[N], prices[N], sigma[N], entry[N], position[N];
/* ... fill returns and prices ... */

mlr_garch model;
if (mlr_garch_fit(returns, TRAIN, &model) != MLR_OK) { /* zero or non-finite variance, or fewer than 100 returns */ }

// sigma[t] is the forecast for period t from returns[0..t-1]; the model's
// stored backcast seeds the recursion, never the data being filtered.
// The fit assumes mean-zero returns: subtract the TRAINING-window mean from
// every return first (the full-sample mean would be a lookahead). A 13%/yr
// drift left in moves alpha by about 0.002; 0.2%/day moves it by 0.015.
if (mlr_garch_filter(&model, returns, N, sigma) != MLR_OK) { /* recursion overflowed */ }

// The same thing for the out-of-sample part alone: continue from the
// variance state at the end of the fit sample. Filtering returns + TRAIN
// with mlr_garch_filter would restart from the backcast instead.
mlr_garch_filter_from(&model, model.sigma2_next, returns + TRAIN, N - TRAIN, sigma + TRAIN);

// A position held over period t is entered at the close of t-1, so it is
// sized against prices[t-1] and its PnL is position[t] * prices[t-1] * returns[t]
for (size_t t = 1; t < N; t++) entry[t] = prices[t - 1];
mlr_vol_target_position(sigma + 1, 0.01, 100000.0, entry + 1, 2.0, N - 1, position + 1);
```

`examples/vol_target_demo.c` runs exactly this inside a walk-forward loop, refitting per fold, and prints the realized volatility of the strategy next to the target. It should land near the target, and it does: over 50 seeds and 4 folds each the ratio averages 1.005 with a standard deviation of 0.067.

Splits, for labels that span 21 periods:

```c
size_t count;
mlr_walk_forward_splits(n, 252, 21, 21, 20, 21, 0, NULL, 0, &count);   // purge = h - 1 = 20
if (count == 0) { /* not enough data for one split */ }
mlr_split *splits = malloc(count * sizeof *splits);
if (mlr_walk_forward_splits(n, 252, 21, 21, 20, 21, 0, splits, count, &count) != MLR_OK) { /* handle */ }
for (size_t i = 0; i < count; i++) {
    // train on [splits[i].train_start, splits[i].train_end)
    // test on  [splits[i].test_start,  splits[i].test_end)
}
free(splits);
```

## How it stays honest

**The forecast at t cannot see t.** `mlr_ewma_vol` emits `sqrt(variance)` before it absorbs `returns[t]`; `mlr_garch_filter` seeds from the model's stored backcast (the mean of squared returns over the fit sample) rather than from the series it is filtering, which is where the 2.x filter leaked. Both are tested for prefix stability (filtering the first half of a series gives the first half of the full filter, exactly) and for shock timing (a spike at bar k moves the output at k+1 and nothing before it). The reference suite goes further: for 40 random series it rewrites everything after a random `t` and asserts every output through `t` is bit-identical.

**The fitter is checked against something it did not write.** `arch` 7.2.0 fits the same simulated sample with the same backcast, so both sides maximize the same function. On the first reference sample mlrisk reaches alpha 0.0810353, beta 0.8835013 and log-likelihood 9259.9490823; `arch` reaches 0.0810352, 0.8835015 and 9259.9490821. Across 20 fresh samples the largest parameter difference is 5e-7. `arch` needs the returns multiplied by 100 to converge on these samples; mlrisk fits them raw, because its Nelder-Mead stops on simplex diameter as well as function value and its feasibility bound on omega is positivity rather than an absolute floor.

**The fitter was checked against brute force.** A 108-start search on the identical likelihood (scipy, adaptive Nelder-Mead) was run over eleven series built to be awkward: near-IGARCH, no ARCH effect, t(3) innovations, a 50-sigma outlier, a fourfold variance regime switch, n = 100, and SPY, BTC and EURUSD. A single Nelder-Mead run from the best grid point lost to it on three of them (by 0.13, 0.02 and 1.7 log-likelihood units: a tiny-alpha series with a second basin at persistence 0.99, the outlier, and the regime switch). The fitter now starts from its three best grid points and restarts each from its own result until that stops helping; it matches the brute-force optimum on ten of the eleven to 1e-10 and beats it by 0.003 on the outlier series, where the maximum sits on the persistence bound and the restarts slide along it.

**Missing data does not poison state.** A NaN return leaves the forecast already made untouched and carries the recursion forward (EWMA keeps its variance, GARCH steps its own one-step forecast). A NaN inside a rolling window makes that window NaN and nothing else. A finite return whose square overflows is treated as missing rather than turning every later sigma into Inf.

**The API does not make the wrong thing easy where it can help it.** Output arrays are declared `restrict` and documented as non-aliasing, because every streaming function reads its input while it writes (passing the same buffer twice corrupts the result, which is the kind of thing that only shows up in production). A leverage cap whose `equity * max_leverage` overflows is rejected rather than silently never applied. And continuing a fitted GARCH onto new data has its own entry point, `mlr_garch_filter_from(model, model.sigma2_next, ...)`, which reproduces the tail of the full filter bit for bit; the plain filter on the new data alone restarts from the backcast and is off by tens of percent for the first few dozen periods, so the header says so.

**Bad arguments refuse the call; bad elements are handled per function, and the rule is written down.** Non-finite scalars (`target_vol`, `equity`, `max_leverage`, `lambda`, `ridge`, `max_dd`, `fraction`, `sigma2_first`) return `MLR_EINVAL`. What a non-finite element does is one of five documented policies, tabulated under Conventions.

**The fit, the likelihood and the filter share one recursion step.** `omega + alpha*r*r + beta*s2` and `omega + alpha*(r*r) + beta*s2` differ by an ulp on about a third of steps, and when the fit computed `sigma2_next` one way and the filter stepped the other, the documented bit-exact continuation failed for 12% of fit samples (24 of 200) by one ulp that then propagated. One `garch_step` function now serves all three, and the continuation is checked across 60 fits in C and 60 more in the reference suite.

**The tests were mutation-tested.** Eleven deliberate breakages (drop the ridge term, drop the offset shift, drop the predict dimension check, drop each overflow guard, revert the optimizer criterion, revert the EWMA alignment, and so on) were compiled against the suite; ten failed at least one assertion and the eleventh is unreachable through the public API because inputs are validated before the solver sees them.

**Every function is fed garbage on every run.** `tests/test_fuzz.c` runs 4000 rounds, 13 to 18 calls each, covering the whole API with random sizes and contents (NaN, Inf, denormals, 1e308, negative zero, `SIZE_MAX` arguments) under AddressSanitizer and UBSan in CI, and checks the promises rather than the numbers: no crash, only documented status codes, positions finite and under the cap, filter output never Inf, a failed fit leaves the model unfitted. Every fourth round is clean so the functions that demand all-finite input are reached on their success paths too (676 GARCH fits, about a thousand each of Kelly and ridge, per run); the first version of the sweep never once fitted a GARCH model, because an all-finite draw of a hundred values had probability 9e-6. Its findings so far: a denormal price made `equity / price` overflow past the leverage cap, and a hand-built model with omega near 1e308 made the filter emit Inf with `MLR_OK`. Both fail closed now (zero position; `MLR_EDOMAIN`).

## Real data

`tests/reference/real_data_check.py` runs the same machinery over daily OHLC files. Six daily series from Yahoo, each from its first listing or 2000 through September 2026, and one month of 1-minute BTC perpetual bars:

| Series | Bars | Inconsistent bars | alpha | beta | alpha + beta | alpha vs arch | Realized vol / target |
|---|---|---|---|---|---|---|---|
| SPY | 6707 | 0 | 0.151 | 0.808 | 0.959 | 1.1e-8 | 1.03 |
| AAPL | 6707 | 0 | 0.093 | 0.871 | 0.964 | 3.3e-9 | 1.04 |
| TLT | 6063 | 0 | 0.085 | 0.893 | 0.978 | 5.5e-9 | 1.05 |
| GLD | 5481 | 0 | 0.109 | 0.865 | 0.974 | 6.8e-8 | 1.03 |
| EURUSD | 5906 | 128 | 0.048 | 0.938 | 0.986 | 2.7e-7 | 1.01 |
| BTC-USD | 4370 | 0 | 0.091 | 0.873 | 0.964 | 4.0e-8 | 0.92 |

Fits are on the last 2000 demeaned log returns; the log-likelihood is never below `arch`'s and the fit on returns multiplied by 100 agrees with the raw fit to 3e-8 in persistence. The 128 inconsistent EURUSD bars (open or close outside the day's range, a quirk of that feed) produce exactly 128 Garman-Klass NaNs and nothing else. The filters are prefix-stable on every series, bit for bit.

The 1-minute BTC series (45,030 returns, rms 1e-3) is the case that broke 2.x: on the raw returns, on returns scaled by 1e-4 (rms 1e-7, omega 1.5e-16) and by 1e4, mlrisk returns alpha 0.100410 and beta 0.893285 every time. `arch` on the same sample scaled by 100 stops at its starting values (0.10, 0.88), 50 log-likelihood units short, and reports success; scaled by 1000 it reaches mlrisk's numbers to six decimals.

## Performance

`bench/bench.c` (configure with `-DMLRISK_BUILD_BENCH=ON`, run `./build/mlrisk_bench`). Apple M-series, clang, `-O3`, best of 5:

| Function | n | Time | ns per element |
|---|---|---|---|
| `mlr_rolling_mean`, window 50 | 10,000,000 | 49 ms | 4.9 |
| `mlr_rolling_std`, window 50 | 10,000,000 | 205 ms | 20.5 |
| `mlr_ewma_vol` | 10,000,000 | 39 ms | 3.9 |
| `mlr_garch_filter` | 10,000,000 | 49 ms | 4.9 |
| `mlr_garch_fit` | 100,000 | 506 ms | three starts, about 1400 likelihood evaluations |

Each streaming row is within 10% of ten times the row for a tenth of n. The fit is linear in n because a Nelder-Mead start takes about 150 iterations whatever the sample size; a run on iid returns (no ARCH effect, alpha at the boundary) used to hit the 2000-iteration cap because the convergence test was relative to alpha itself, which is one of the 3.1.0 fixes. The three starts and their restarts cost 5 to 7x a single run, which is 8 ms at n = 1000.

## Conventions

Volatility is **per period** everywhere. Annualized to per-period: divide by `sqrt(periods_per_year)` (252 daily, 52 weekly, 12 monthly).

`mlr_rolling_std` is population variance (divides by `window`; pandas `rolling().std()` defaults to the sample convention). `mlr_kelly_fraction` uses sample variance (`n-1`) and the raw mean, not the excess over a funding rate. `mlr_garch_fit` is Gaussian MLE on mean-zero returns, so demean with the training-window mean before fitting and filtering; the recursion starts from `sigma2[0] = omega + (alpha + beta) * backcast`, the same presample rule as `arch`. Alpha and beta are scale invariant; omega scales with the variance of the returns.

Non-finite elements follow one of five policies, each stated in the function's header:

| Policy | Functions |
|---|---|
| Reject the whole call with `MLR_EINVAL` | `mlr_garch_fit`, `mlr_kelly_fraction`, `mlr_drawdown_scale`, `mlr_linreg_fit` |
| NaN at that index, `MLR_OK` | `mlr_rolling_mean`, `mlr_rolling_std`, `mlr_parkinson_vol`, `mlr_garman_klass_vol` |
| Skip the element, carry the state, forecast unchanged | `mlr_ewma_vol`, `mlr_garch_filter`, `mlr_garch_filter_from` |
| Zero position at that index | `mlr_vol_target_position` |
| Not checked; propagates | `mlr_linreg_predict` |

Three more contracts a caller has to know. `n == 0` is `MLR_EINVAL` everywhere except `mlr_walk_forward_splits`, where it yields zero splits; empty input is not an empty result. Every `size_t` count is trusted, so a negative value converted to `size_t` walks off the buffer. And `backcast == 0` in `mlr_garch` is a sentinel for the unconditional variance, so a hand-built model with the field left at zero seeds from `omega / (1 - alpha - beta)`; `sigma2_next` has no such sentinel and must be set before forecasting. Output arrays must never alias inputs (`MLR_RESTRICT`).

For labels built from `h` periods, the last `h-1` training samples before a test window have labels that overlap it, so pass `purge = h - 1`. The optional post-test training segment runs to the end of the data and therefore contains every later split's test window; it exists for purged-CV model selection and using it for anything the evaluation depends on invalidates that split and every one after it.

Range estimators (`mlr_parkinson_vol`, `mlr_garman_klass_vol`), the rolling statistics (`mlr_rolling_mean`, `mlr_rolling_std`) and `mlr_drawdown_scale` are contemporaneous by construction, index `t` includes period `t`; lag them one bar before sizing. Drawdown scaling is the one most likely to be applied without the lag, because it is a multiplier for position sizes: `scale[t]` is computed from the close at the end of period `t`, so the position held over period `t` gets `scale[t-1]`. Applied without the lag it de-levers on the bar of a loss using that bar's own close. On a consistent bar (open and close inside `[low, high]`) Garman-Klass is never negative, because `|ln(close/open)| <= ln(high/low)` bounds the estimator below by `0.114 * ln(high/low)^2`; inconsistent bars give NaN.

> mlrisk is a set of building blocks, not a backtester and not a portfolio system. It does not know about transaction costs, borrow, calendars, or corporate actions, and the example's PnL is on constant equity with no compounding. `mlr_kelly_fraction` is the textbook mean-over-variance number from a historical sample: an upper bound on sizing, not an allocation model, and it will happily tell you to lever up on a lucky sample. `converged == 1` from the GARCH fitter means the optimizer met its tolerances, not that three parameters are identified by your sample; on 20 observations it will happily converge to a unit root, which is why it insists on 100.

## Validation

`tests/reference/reference_check.py` compiles the C sources into a shared library, calls them through ctypes, and compares against independent implementations. CI runs it on every push. The Python under `tests/reference/` is verification tooling and nothing else: the library has no Python dependency and the C tests do not need it. Measured on the current tree:

| Check | Reference | Result |
|---|---|---|
| Rolling mean and std, windows 1 to n, NaN and Inf gaps | pandas `rolling` | max error 3.1e-11 |
| Rolling std at price level 1e9 | exact rational arithmetic on the input doubles | 1.7e-15 (numpy two-pass: 4.9e-12) |
| Rolling mean and std after a bad first tick of 1e9 to 1e15, and along a trend 100 to 1e6 | exact rational arithmetic | 6e-16 std, exact mean |
| EWMA, predictive alignment | pandas `ewm(adjust=False)` shifted one period | 3.5e-18 |
| GARCH filter and 20-step forecast | `arch` conditional variance and `forecast()` | 4.4e-16 and 8.9e-16 relative |
| GARCH fit, 20 samples, same likelihood | `arch` | max parameter difference 5.0e-7 |
| Parkinson and Garman-Klass, inconsistent bars included | numpy | 7.9e-16 |
| Kelly, drawdown scaling, vol targeting with cap | numpy | 4.4e-15, exact, 4.6e-13 |
| Ridge, d in 1..8, ridge 0..10, features 1e-6..1e6 | scikit-learn and closed form | 8.3e-15 of std(y) |
| Ridge on designs with condition number 1e4 to 1e10 | SVD least squares | 1x cond times epsilon (2.8e-8 at 1e8) |
| Ridge slope with a feature at level 1e6 to 1e15 | exact rational OLS | under 1e-14 (scikit-learn: 1.7e-5 at 1e15) |
| Walk-forward splits, 1620 parameter sets | independent generator | 0 mismatches |
| Purge rule, labels spanning 2, 5 and 21 periods | leakage counted by construction | purge = h-1 clean, h-2 leaks |
| No lookahead, 40 trials, 4 functions | bitwise prefix comparison | 0 violations |
| Vol-targeting loop, 50 seeds, 4 folds | realized vol / target | mean 1.005, sd 0.067 |
| Example binary PnL, seed 42 | recomputed from raw arrays | within 0.004 currency units |
| GARCH continuation from `sigma2_next`, 60 fits | tail of the full filter | bit-identical |

GARCH(1,1) parameter recovery, 200 simulated series per row, truth alpha 0.10, beta 0.85:

| n | alpha bias | alpha rmse | beta bias | beta rmse | persistence rmse | converged |
|---|---|---|---|---|---|---|
| 500 | +0.0033 | 0.035 | -0.036 | 0.100 | 0.084 | 200/200 |
| 2000 | +0.0012 | 0.017 | -0.007 | 0.028 | 0.018 | 200/200 |

The beta bias at n = 500 is the estimator, not the code (`arch` lands on the same values); it is the reason the fitter refuses fewer than 100 observations and the reason to distrust short fits even when it does not refuse.

## API

The headers are the documentation.

| Function | Header | Purpose |
|---|---|---|
| `mlr_rolling_mean`, `mlr_rolling_std` | `rolling.h` | Trailing-window statistics, O(n) |
| `mlr_ewma_vol` | `rolling.h` | RiskMetrics EWMA volatility forecast |
| `mlr_garch_fit`, `mlr_garch_filter`, `mlr_garch_filter_from`, `mlr_garch_forecast` | `vol.h` | GARCH(1,1) by MLE, filter any series, continue from a variance state, multi-step forecast |
| `mlr_parkinson_vol`, `mlr_garman_klass_vol` | `vol.h` | Per-bar range estimators |
| `mlr_vol_target_position` | `sizing.h` | Volatility targeting with a notional cap |
| `mlr_kelly_fraction` | `sizing.h` | Mean-variance Kelly fraction |
| `mlr_drawdown_scale` | `sizing.h` | Linear exposure scaling by drawdown |
| `mlr_walk_forward_splits` | `split.h` | Purged, embargoed walk-forward splits |
| `mlr_lin_model_init`, `mlr_linreg_fit`, `mlr_linreg_predict`, `mlr_lin_model_free` | `linreg.h` | Ridge regression |
| `mlr_version`, `mlr_version_number`, `MLRISK_VERSION` | `version.h` | Version at runtime (for bindings that load the binary) and as macros |
| `MLR_GARCH_MIN_N`, `MLR_GARCH_MAX_PERSISTENCE` | `vol.h` | The fitter's sample floor and the persistence bound, as constants a caller can check against |

Every function returns `mlr_status`: `MLR_OK`, `MLR_EINVAL`, `MLR_ENOMEM`, `MLR_EBOUNDS` (split capacity too small; the required count is still reported), `MLR_EDOMAIN` (singular system, zero variance, overflow, non-positive equity).

## Changelog

[CHANGELOG.md](CHANGELOG.md). 3.0.0 unified the timing convention and was a breaking release; 3.1.0 is the verification pass that produced most of the numbers above; 3.2.0 replaced the regression solver with QR and closed the extreme-input and misuse paths found in a second review; 3.3.0 is the pre-binding pass: one recursion step, a fitted flag, a version symbol, and the contracts above written down.

## License

MIT, see [LICENSE](LICENSE).
