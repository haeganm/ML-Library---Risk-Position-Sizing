# Changelog

## 3.0.0 (2026-09-03)

Breaking release. Every volatility forecast now shares one timing convention,
and a technical audit closed a set of silent-failure paths.

### Breaking

- `mlr_ewma_vol` is predictive: `out[t]` is the forecast for period t from
  `returns[0..t-1]`, matching `mlr_garch_filter`. Previously `out[t]` included
  `returns[t]`, so sizing period t from it was a one-bar lookahead. `out[0]` is
  now `NAN`. `lambda` must be in `[0, 1)`.
- `mlr_lin_model_init(model, d)` drops the ridge argument; the `ridge` field is
  written by `mlr_linreg_fit` with the value actually used (it was previously
  never updated and could disagree with the fit).
- `mlr_garch_fit` requires `n >= 100` (was 20).
- `mlr_garch` gained a `backcast` field (appended). The recursion starts from
  `sigma2[0] = omega + (alpha + beta) * backcast`, the `arch` presample rule.
  Fitted parameters move by about 1e-5 relative to 2.0.0.
- CMake 3.21 is required; the library compiles as strict ISO C11.

### Fixed

- `mlr_garch_filter` seeded its recursion from the series being filtered, so
  every output depended on future returns. The seed now comes from the model.
- `mlr_garch_fit` had an absolute feasibility floor on `omega`; returns with
  rms below about 2e-8 returned an unfitted model with `MLR_OK`. The bound is
  now positivity only and estimates are scale invariant.
- Nelder-Mead stopped on function-value spread alone, so parameter accuracy
  varied about 1000x with the units of the returns. Convergence now also
  requires a small simplex; accuracy is about 1e-7 at any scale.
- `mlr_garch_filter` and `mlr_garch_forecast` accepted `omega = Inf` from
  hand-built models and produced `Inf`/`NaN` output with `MLR_OK`.
- `mlr_walk_forward_splits` could wrap `size_t` arithmetic: a huge `embargo`
  placed the post-test training segment inside the test window, and a huge
  `train_len` produced out-of-range splits, both with `MLR_OK`.
- `mlr_linreg_fit` accepted NaN or Inf inputs (NaN pivots passed the
  singularity test) and negative ridge, producing garbage models with
  `MLR_OK`. Its singularity threshold had an absolute floor that rejected
  well-conditioned systems with features below about 1e-8.
- `mlr_linreg_predict` dereferenced a freed model.
- `mlr_vol_target_position` did not check `equity` or `max_leverage` for
  finiteness: NaN equity gave NaN positions, Inf leverage disabled the cap.
- `mlr_kelly_fraction` could return NaN with `MLR_OK` on overflowing sums.
- Missing data no longer poisons `mlr_ewma_vol` or `mlr_garch_filter`; the
  forecast already made is kept and the recursion carries on.
- Range estimators return `NAN` rather than `Inf` when the price ratio overflows.

### Added

- `mlr_garch_fit` is checked against the Python `arch` package on identical
  samples with an identical likelihood (`tests/reference/garch_arch_reference.py`).
- Tests for prefix stability of the GARCH filter, scale invariance of the fit,
  the timing alignment of every forecast, ridge shrinkage, and all of the
  input-validation paths above. Tests run per module under ctest.
- `mlrisk/version.h` (`MLRISK_VERSION`), a pkg-config file, and CMake options
  `MLRISK_WERROR`, `MLRISK_BUILD_TESTS`, `MLRISK_BUILD_EXAMPLES` for consumers.
- The example runs a walk-forward GARCH fit, filter, and sizing loop and
  reports realized strategy volatility against the target.

## 2.0.0

Public API rework: `mlr_` prefix on every function, purged and embargoed
walk-forward splits with a capacity-safe API, GARCH(1,1), range estimators,
Kelly and drawdown sizing. No compatibility layer with 1.x.
