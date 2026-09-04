# mlrisk

> A C11 library for volatility forecasting, position sizing, and leakage-safe walk-forward evaluation.

[![CI](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml/badge.svg)](https://github.com/haeganm/mlrisk/actions/workflows/ci.yml)
[![C11](https://img.shields.io/badge/C-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- **Volatility forecasts**: EWMA, GARCH(1,1) fitted by maximum likelihood, and per-bar Parkinson and Garman-Klass range estimators
- **Position sizing**: volatility targeting with a leverage cap, Kelly fractions, drawdown-based exposure scaling
- **Walk-forward splits** with purging and embargo, so labels never overlap the test window
- **Rolling mean and standard deviation** in O(n), stable at large price levels
- **Ridge regression** for small feature sets

Zero dependencies beyond libm. Builds as strict ISO C11 on GCC, Clang, and MSVC, with tests on Linux, macOS, and Windows plus an AddressSanitizer/UBSan job. Every volatility forecast shares one timing convention (below), and the GARCH fitter is checked against the Python `arch` package on identical samples.

This is research software, not investment advice.

## Build

Requires CMake 3.21 and a C11 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/vol_target_demo            # Windows: .\build\Release\vol_target_demo.exe
```

Use it from CMake either as a subdirectory or after `cmake --install build`:

```cmake
add_subdirectory(path/to/mlrisk)          # or: find_package(mlrisk 3 REQUIRED)
target_link_libraries(your_target PRIVATE mlrisk::mlrisk)
```

A pkg-config file (`mlrisk.pc`) is installed too. Options: `MLRISK_WERROR`, `MLRISK_BUILD_TESTS`, `MLRISK_BUILD_EXAMPLES` (all default ON when mlrisk is the top-level project, OFF when consumed).

## Usage

```c
#include "mlrisk/mlrisk.h"

// returns[t] is the return over period t; prices[t] the close of period t
mlr_garch model;
if (mlr_garch_fit(returns, n_train, &model) != MLR_OK) { /* handle */ }

// sigma[t] is the forecast for period t, made from returns[0..t-1]
double sigma[N];
mlr_garch_filter(&model, returns, N, sigma);

// A position held over period t is entered at the close of t-1, so size it
// against prices[t-1]. Its PnL is position[t] * prices[t-1] * returns[t].
double position[N];
mlr_vol_target_position(sigma + 1, 0.01 /* target vol per period */,
                        100000.0 /* equity */, prices /* prices[t-1] for t>=1 */,
                        2.0 /* max leverage */, N - 1, position + 1);
```

`examples/vol_target_demo.c` runs this end to end inside a walk-forward loop and reports realized strategy volatility against the target.

Purged walk-forward splits:

```c
// 252-period training windows, 21-period test windows, stepping by 21.
// Labels are 21-period forward returns, so purge 20 (h - 1) training samples.
size_t count;
mlr_walk_forward_splits(n, 252, 21, 21, 20, 0, 0, NULL, 0, &count);
mlr_split *splits = malloc(count * sizeof *splits);
mlr_walk_forward_splits(n, 252, 21, 21, 20, 0, 0, splits, count, &count);
for (size_t i = 0; i < count; i++) {
    // train on [splits[i].train_start, splits[i].train_end)
    // test on  [splits[i].test_start,  splits[i].test_end)
}
free(splits);
```

## Conventions

**Timing.** `mlr_ewma_vol` and `mlr_garch_filter` are predictive: output `t` is the forecast for period `t` from returns before `t`. Position `t` from `mlr_vol_target_position` is held over period `t` and scored against `returns[t]`. Nothing at index `t` has seen `returns[t]`. The range estimators are per-bar and contemporaneous by construction; lag them one bar before sizing. Missing data (a non-finite return) never changes a forecast already made; the recursions carry on and recover.

**Per-period volatility** everywhere. Annualized to per-period: divide by `sqrt(periods_per_year)` (252 daily, 52 weekly, 12 monthly).

**Variance.** `mlr_rolling_std` is population variance (divides by `window`; pandas `rolling().std()` defaults to sample). `mlr_kelly_fraction` uses sample variance (`n-1`). `mlr_garch_fit` is Gaussian MLE on mean-zero returns, with the recursion started from `sigma2[0] = omega + (alpha + beta) * backcast`, `backcast = mean(r^2)` over the fit sample, the same presample rule as `arch`.

**Purging.** For labels built from `h` periods, the last `h-1` training samples before a test window overlap it; pass `purge = h - 1`. The optional post-test training segment is future data relative to the test window and contains every later split's test window; it is for purged-CV model selection only.

**Errors.** Every function returns `mlr_status`: `MLR_OK`, `MLR_EINVAL` (bad argument), `MLR_ENOMEM`, `MLR_EBOUNDS` (output capacity too small; splits only), `MLR_EDOMAIN` (singular system, zero variance, overflow, bad equity path). Bad *elements* (a NaN price, a bar with high < low) produce a NaN or zero at that index with `MLR_OK`; bad *arguments* fail the call. `mlr_isfinite` and `mlr_isnan` are exported for callers.

## API

Full documentation lives in the headers.

| Function | Header | Purpose |
|---|---|---|
| `mlr_rolling_mean`, `mlr_rolling_std` | `rolling.h` | O(n) trailing-window statistics |
| `mlr_ewma_vol` | `rolling.h` | RiskMetrics EWMA volatility forecast |
| `mlr_garch_fit`, `mlr_garch_filter`, `mlr_garch_forecast` | `vol.h` | GARCH(1,1) by MLE; filter any series; multi-step forecast |
| `mlr_parkinson_vol`, `mlr_garman_klass_vol` | `vol.h` | Per-bar range estimators |
| `mlr_vol_target_position` | `sizing.h` | Volatility targeting with a leverage cap |
| `mlr_kelly_fraction` | `sizing.h` | Mean-variance Kelly fraction |
| `mlr_drawdown_scale` | `sizing.h` | Linear exposure scaling by drawdown |
| `mlr_walk_forward_splits` | `split.h` | Purged, embargoed walk-forward splits |
| `mlr_lin_model_init`, `mlr_linreg_fit`, `mlr_linreg_predict`, `mlr_lin_model_free` | `linreg.h` | Ridge regression |
| `MLRISK_VERSION` | `version.h` | Version macros |

## Validation

The GARCH fitter is tested against `arch` 7.2.0 on two simulated samples. Both sides fit the identical series with the identical backcast, so they maximize the same likelihood; the constants are regenerated by `tests/reference/garch_arch_reference.py`.

| Sample 1 (n = 2000) | mlrisk | arch |
|---|---|---|
| alpha | 0.0810353 | 0.0810352 |
| beta | 0.8835013 | 0.8835015 |
| omega | 1.355614e-6 | 1.355607e-6 |
| log-likelihood | 9259.9490823 | 9259.9490821 |

The second sample (persistence 0.976) agrees to the same precision. `arch` needs the returns rescaled by 100 to converge on these samples; mlrisk fits the raw series, and its estimates are invariant to the scale of the input.

The rolling standard deviation is checked against a two-pass computation at a price level of 1e9 to 1e-12, and every forecast is checked for prefix stability (the output at `t` is identical whether or not the data after `t` is present).

## Changelog

See [CHANGELOG.md](CHANGELOG.md). 3.0.0 is a breaking release; the EWMA alignment and the linreg init signature changed.

## License

MIT, see [LICENSE](LICENSE).
