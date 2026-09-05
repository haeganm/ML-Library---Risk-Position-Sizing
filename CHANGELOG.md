# Changelog

## Unreleased

### Fixed

- `mlr_garch_fit` could stop at a local maximum on a short sample with no
  ARCH effect. Its seed grid had no start below beta = 0.8, and from any
  high-persistence seed Nelder-Mead settled at alpha = 0 with beta near 1,
  0.2 log-likelihood units short of the maximum at beta = 0, and reported
  convergence. The grid now includes beta = 0 and 0.5 and one extra start
  is taken from its best low-persistence point, kept separate from the
  three high-persistence starts because on a series with a large outlier
  the low seeds score well on the grid and would otherwise crowd out the
  starts that find the right answer. Fit time is up by a third (621 ms for
  100,000 returns). Verified by the new
  `tests/reference/garch_multistart_check.py`, 108 starts against the fit
  on thirteen awkward series, and by a C test over 40 short iid samples.
  Well-behaved samples fit to the same parameters as before to seven
  decimals.
- The Python binding read a masked array through its fill values: the mask
  was dropped by the float64 conversion, so a masked observation was a real
  number to the C. Masked entries are now NaN, which every function treats
  as missing.
- The binding accepted a scalar, `None`, or a datetime series as input
  because the rank check ran after the conversion that promoted or recast
  them. A scalar was a one-element series and a datetime became
  nanoseconds. Both are refused now, before conversion.
- Paired pandas inputs (`parkinson_vol`, `garman_klass_vol`,
  `vol_target_position`) were matched by position and the result stamped
  with the first argument's index, so two Series of the same length on
  different indexes were silently mispaired. Series arguments must now
  share an index; `lag` keeps it, slicing does not.
- A non-integer count such as `inf` raised `OverflowError` instead of the
  documented `TypeError`; an explicit `purge=` that was too large produced
  an error blaming a label horizon the caller never passed; the
  `PurgedWalkForward` docstring claimed every observation after the first
  training window is tested, which is false when the sample does not end
  on a whole test window.
- The source distribution excluded `examples/` and `bench/` but kept the
  root `CMakeLists.txt`, which builds both when the library is the
  top-level project, so `cmake -S . -B build` on an unpacked sdist failed
  at configure. The sdist now carries everything that file builds.
- The notebook's CI step only checked that no cell raised. The notebook
  now asserts the numbers the README quotes, so a change in the library
  that moves them fails the build.

### Added

- `examples/spy_walk_forward.ipynb`: one model end to end on 25 years of
  daily SPY (checked in as `examples/data/SPY.csv`), with lagged features,
  a ridge on a five-day forward return, 41 purged walk-forward folds, a
  GARCH per fold continued with `filter_from`, and 10% volatility
  targeting. It reports the honest out-of-sample number, then shows what
  skipping the purge and dropping the lags each do to it, against a
  control that changes the training rows by the same amount and leaks
  nothing.
- `py.typed`, so type checkers use the package's annotations. Metadata 2.4
  with a license expression, Python version classifiers, source and issue
  URLs.
- The release workflow's check job now requires five wheels at the tagged
  version, each carrying the shared library, and a changelog section for
  the version; publishing tolerates a re-run after a partial upload.
  Dependabot watches the actions.

### Changed

- The README's precision figures for the rolling statistics were stale
  (1.7e-15 and 6e-16 where the reference suite measures 4.4e-16 and
  4.1e-15), the 3.4.0 entry claimed a glibc 2.17 floor the wheels do not
  have, and the purge's effect on the SPY example's Sharpe was attributed
  to the four leaked rows when a leak-free shift of the same size moves it
  as much. All three now say what the measurements say. The brute-force
  check the README described lives in the repository instead of being a
  claim.
- The wheel's C is compiled with `-Wpedantic` like the library.
- `GarchModel.filter_from` docstring says that the seed is a variance and
  the output is volatility, which the C header already said and the
  example's first draft got wrong; `GarchModel.forecast` no longer
  describes a variance decaying toward a volatility.

## 3.4.1 (2026-09-05)

No code change. This release exists to publish through trusted publishing
for the first time; 3.4.0 went out with an API token that has since been
revoked. The release workflow, `RELEASING.md` and the 3.4.0 entry below now
describe what actually happened.

## 3.4.0 (2026-09-04)

First release published to PyPI as `walkforward`. The C library is unchanged
from 3.3.1; everything here is the Python package and the machinery to ship
it. The first upload went through an API token after a trusted-publishing
attempt was refused; the cause was a one-letter mismatch in the publisher's
workflow name (`release.yaml` for a file called `release.yml`), not the
platform incompatibility this entry briefly claimed. Trusted publishing is
the configured path from here on.

### Added

- A Python package, `walkforward`, in `python/`. It binds this library with
  ctypes and numpy: `ewma_vol`, `garch_fit` returning a `GarchModel` with
  `persistence`, `half_life` and `unconditional_vol`, the rolling statistics
  with a `lag` helper, the sizing functions, `Ridge`, and
  `PurgedWalkForward`, a scikit-learn cross-validator.
- Two API decisions worth naming. The splitter asks for `label_horizon`
  rather than a purge count, because `purge = h - 1` is the part users get
  wrong, and it has no `embargo` argument, because a walk-forward never
  trains on data after the window it is testing. `walk_forward_splits`
  exposes the post-training variant for people who want it.
- The binding allocates every output buffer itself, so the `restrict`
  non-aliasing contract on the C outputs cannot be violated by a numpy view,
  and a pandas Series keeps its index through a call.
- Wheels are `py3-none`: the binding never touches the Python C API, so one
  wheel per platform serves every Python 3.
- A `python-package` CI job builds and tests the wheel on Linux, macOS and
  Windows, asserts the package and the library it bundles report the same
  version, and installs from a source distribution in a clean directory so
  the sdist keeps carrying the C it has to compile.
- The Python packaging sits at the repository root rather than in `python/`,
  because a source distribution has to contain `src/` and `include/` and
  those cannot be reached from a subdirectory.
- A `Release` workflow that builds the sdist and five wheels (Linux x86-64
  and aarch64, macOS Intel and Apple silicon, Windows x64), tests each wheel
  on the platform it was built for, installs the sdist from scratch, and
  publishes to PyPI through trusted publishing, with no API token in the
  repository. A tag that disagrees with the version in `CMakeLists.txt` is
  refused before anything is built. See `RELEASING.md`.
  Validated by running the workflow before any tag: it produces one
  `py3-none` wheel per platform, each carrying its own shared library. The
  Linux wheels are built on manylinux_2_28 and tagged manylinux_2_17,
  since the library links nothing newer; numpy sets the real glibc floor.

### Changed

- The README is rewritten around the Python package, which is how most people
  will use this, with the C library as its own section rather than the
  headline. Every measured number is carried over, and both Python snippets
  in it are executed as part of the checks.
- The repository is now `haeganm/walkforward`, matching the Python package.
  GitHub redirects the old address and the earlier releases are unaffected.
  The C keeps its `mlr_` prefix and its `include/mlrisk/` headers, the way
  Pillow ships `PIL`.


## 3.3.1 (2026-09-04)

- `mlr_drawdown_scale` is contemporaneous (`scale[t]` uses the close at the
  end of period `t`) and its header said "multiply position sizes by
  scale_out" with no lag, in the same file whose timing contract says index
  `t` must not know period `t`. Applying `scale[t]` to the position held
  over period `t` de-levers on the bar of a loss using that bar's own close.
  The header, the sizing contract and the README now say to use
  `scale[t-1]`; a test pins the alignment. No code change.
- The rolling mean and standard deviation shifted every value by the first
  finite value of the whole series and never revisited that choice. A bad
  first tick therefore corrupted every later window: with prices near 100
  and a first value of 1e9, windows that did not even contain it had a
  standard deviation 600% off and a mean off by 1e-6; a first value of 1e12
  made the std meaningless. A long trend away from the starting level eroded
  precision the same way. The offset now comes from inside the current
  window, the accumulators are rebuilt every `window` samples (amortized
  O(1)) and whenever a leaving sample carried almost all of the variance or
  the sum, so an outlier that has left leaves no rounding behind. Measured
  against exact rational arithmetic: 4.1e-15 relative for the std and an
  exact mean after first ticks of 1e9, 1e12 and 1e15 and along a trend from
  100 to 1e6 (this entry first said 6e-16; the reference suite has always
  measured 4.1e-15 on this code). Cost: about 6 ns per element for the mean and 20 for the std,
  up from 4 and 15.
- `mlr_linreg_fit` centered each feature with a plain running sum, so a
  feature at a large level carried rounding of order n * eps * level into
  every centered value: with a price column near 1e9 and unit variation
  the slope was off by 4e-12, and at 1e12 by 7e-7, while an SVD solve
  reached 1e-15. Columns are now shifted by the first row before centering
  (differences of nearby doubles are exact), which puts the slope at 4e-15
  from 1e6 to 1e15, ahead of scikit-learn's 1.7e-5 at 1e15.
- The rolling mean returned `Inf` and the rolling std returned `0` with
  `MLR_OK` for a window of finite values whose differences overflow (values
  of opposite sign near 1e308): the shifted accumulators overflowed and the
  `fmax` that guards an epsilon-negative variance turned the resulting NaN
  into a zero standard deviation. Such windows are now NaN, the state is
  rebuilt on the next step, and no output is ever Inf.
- `mlr_garch_fit` documented `MLR_EDOMAIN` for a denormally small variance
  but only rejected zero; at a backcast of 4e-321 it returned `MLR_OK` with
  alpha drifted by 0.003 and `converged = 0`. Variances below `DBL_MIN` are
  now refused as documented; anything above fits to the same parameters as
  the unscaled series.
- `mlr_kelly_fraction` returned `f = 0` with `MLR_OK` when the squared
  deviations overflowed (variance `Inf`), a silently rounded estimate where
  the same function already refuses the NaN form of the overflow. It now
  returns `MLR_EDOMAIN`.


## 3.3.0 (2026-09-04)

The last pass before the C API is frozen for language bindings. Three
reviewers who had not seen the code read it; nine of their findings
reproduced and are fixed here.

### Fixed

- The GARCH recursion was written three ways: the likelihood and the fit's
  `sigma2_next` loop as `omega + alpha*r*r + beta*s2`, the filter as
  `omega + alpha*(r*r) + beta*s2`. The two differ by an ulp on about a third
  of steps, so for 12% of fit samples `sigma2_next` was not the value the
  filter reached and the documented bit-exact `mlr_garch_filter_from`
  continuation failed by an ulp. One `garch_step` now serves all three;
  the continuation is tested across 60 fits.
- `mlr_garch_forecast` accepted `sigma2_next == 0` (the value in a
  hand-built model that never set it) and forecast zero volatility with
  `MLR_OK`. Now `MLR_EINVAL`, matching `mlr_garch_filter_from`.
- `mlr_linreg_fit` wrote weights into the model during back-substitution
  and the intercept before checking it, so a failing fit left a half-new
  model behind. It now solves into scratch and touches the model only on
  success; the header says "on any failure the model is left exactly as it
  was".
- `mlr_linreg_predict` on a model that was initialized but never fitted
  returned all zeros with `MLR_OK`. `mlr_lin_model` gained a `fitted` field
  (set by a successful fit, cleared by init and free) and predict requires
  it. Recompile consumers: the struct grew.
- The installed `mlrisk.pc` located the prefix as `${pcfiledir}/../..`,
  which is wrong when `CMAKE_INSTALL_LIBDIR` is two levels deep
  (`lib/x86_64-linux-gnu`, the Debian and Ubuntu default under `/usr`).
  The relative path is now computed at configure time.
- `mlr_walk_forward_splits` did not write `*count_out` on its `MLR_EINVAL`
  paths; it now writes 0.
- The fuzz sweep never fitted a GARCH model: every element was non-finite
  with probability 0.11 and the fit needs 100 finite returns, so the
  success path had probability 9e-6 per round. Every fourth round is now
  clean, split parameters can hit every status code, and the sweep asserts
  that the success paths were reached.
- The C11 language requirement was exported to consumers through
  `target_compile_features(PUBLIC)`; the public headers need only C99, so
  it is now private. MSVC builds add `/fp:contract-`, the real counterpart
  of `-ffp-contract=off`. The 32-bit CI leg uses SSE math instead of x87.
- Documentation that no longer matched the code: `linreg.h` described the
  pre-3.2.0 normal-equation solver; `sizing.h`'s lag-before-sizing warning
  omitted the rolling statistics, which are contemporaneous; "with
  ridge > 0 the system is always solvable" was overstated; `mlr_linreg_fit`
  called its in/out model `model_out`.

### Added

- `mlr_version()` and `mlr_version_number()`, so a binding that loads the
  compiled library can check what it loaded (the reference suite now does).
- `MLR_GARCH_MIN_N` and `MLR_GARCH_MAX_PERSISTENCE` as public constants.
- Header contracts a binding author asked for: partially written output on
  `MLR_EDOMAIN`, count-query mode keyed on the pointer, the `backcast == 0`
  sentinel, `converged` exactly 0 or 1, the model owning `w` and not being
  copyable, `mlr_ewma_vol` all-NaN when no usable return precedes the last
  index, the lag warning on each range estimator. README Conventions now
  tabulates the five non-finite-element policies and states the `n == 0`
  and trusted-`size_t` rules.
- `MLRISK_BUILD_BENCH` option (default off); the benchmark is no longer
  built with the examples.
- A reference check that the purge rule is exactly the leakage boundary:
  with labels spanning h periods, purge = h-1 leaves no training label
  inside the test window and purge = h-2 does, for h = 2, 5 and 21.
- The README says to demean with the training-window mean before fitting
  (the full-sample mean is a lookahead) and quantifies the drift bias.

### Changed

- Size-overflow rejections in `mlr_lin_model_init` and `mlr_linreg_fit`
  return `MLR_EINVAL` (a dimension that cannot be sized is bad input);
  `MLR_ENOMEM` is reserved for a real allocation failure.
- `tests/reference/requirements.txt` pins `statsmodels`; the reference
  build uses `-O3` to match the CMake Release build.
- Work arrays in `mlr_linreg_fit` are zero-initialized; GCC's analyzer
  flagged reads it could not prove written (they were), and removing the
  question costs nothing next to the solve.


## 3.2.0 (2026-09-04)

A second review (by a reader who had not seen the code before) named six
weak spots. Four were real bugs or traps and are fixed; the other two were
design choices and are now either better or better documented.

### Fixed

- `mlr_linreg_fit` sized its work arrays with a constant chosen for 64-bit
  `size_t`; on 32-bit targets the `d * (d + 1)` product wrapped for d above
  about 23,000 and the fit could write past its allocation. Sizes are now
  checked arithmetically before anything is allocated or read.
- `mlr_vol_target_position` accepted an `equity` and `max_leverage` whose
  product overflowed, which made the cap infinite and therefore never
  applied. Such a pair is now `MLR_EINVAL`.
- Passing the same array as input and output corrupted rolling and EWMA
  results. Output parameters are now `restrict`-qualified (`MLR_RESTRICT`,
  empty under C++) and the contract is documented in `types.h`.
- Continuing a fitted GARCH model onto new data by filtering the new data
  alone restarted the recursion from the pre-sample backcast (about 30% off
  at the first period on the test series). `mlr_garch_filter_from` starts
  from a given variance, and with `model->sigma2_next` reproduces the tail
  of the full filter bit for bit; the example uses it and the plain filter's
  header explains the trap.

### Changed

- The regression solver is Householder QR on the centered design with
  `sqrt(ridge) I` appended, instead of Gaussian elimination on the normal
  equations. Accuracy is now about condition number times epsilon (2.8e-8
  at condition number 1e8, where the old solver refused the problem and was
  already at 1.5e-5 by 1e6). Same API, same results on well-conditioned data
  to 1e-14.
- `mlr_kelly_fraction` is documented as a sizing utility and an upper bound,
  not an allocation model.

### Added

- Reference checks for the QR solver on designs with condition numbers
  1e4 to 1e10 against SVD least squares, and for `mlr_garch_filter_from`
  continuation; C tests for the ill-conditioned fit, the cap overflow, the
  continuation, and the 32-bit size guard.


## 3.1.0 (2026-09-03)

Verification release. Every public function is now compared against an
independent implementation on every push, and the pass turned up a handful
of contract gaps.

### Fixed

- The GARCH optimizer's convergence test measured the simplex relative to
  each parameter's own magnitude, so a maximum on the boundary `alpha = 0`
  (returns with no ARCH effect) could never satisfy it: the fit ran to the
  2000-iteration cap, took 16x longer than it should, and reported
  `converged = 0` on a finished fit. The simplex is now measured against a
  fixed scale per parameter (the backcast for omega, 1 for alpha and beta).
- `mlr_garch_fit` ran Nelder-Mead once, from the best grid point, and could
  stop in the wrong basin: against a 108-start brute-force search on the
  identical likelihood it lost on three of eleven adversarial series (tiny
  alpha with a second maximum at persistence 0.99, a 50-sigma outlier, a
  fourfold variance regime switch), by up to 1.7 log-likelihood units. It now
  runs from its three best grid points, restarts each from its own result
  until that stops improving, and keeps the best; it matches the brute-force
  optimum on ten of the eleven and beats it on the outlier series. Fits cost 5 to 7x what they did (8 ms at n = 1000).
- `mlr_vol_target_position` let a denormal price overflow `equity / price`
  past the leverage cap (an infinite position with `MLR_OK`), and let a
  denormal position slip the cap by rounding; both are now zero.
- `mlr_garch_filter` and `mlr_garch_forecast` emitted `Inf` when extreme but
  valid parameters overflowed the recursion; they now return `MLR_EDOMAIN`.
- `mlr_lin_model_init` and `mlr_linreg_fit` refuse dimensions whose
  allocation size would overflow (`MLR_ENOMEM`) instead of relying on
  `calloc` to notice.
- `mlr_garch_filter` treated a finite return whose square overflows as data,
  pinning every later sigma at `Inf` with `MLR_OK`; it is now treated as
  missing, the same rule EWMA already used.
- `mlr_garman_klass_vol` accepted bars with the open or close outside
  `[low, high]`. They now give `NAN`. On a consistent bar the estimator is
  bounded below by `0.114 * ln(high/low)^2`, so the "negative variance"
  branch was unreachable and has been removed from the code and the docs.
- The installed `mlrisk.pc` hard-coded the configure-time prefix, so an
  install with `--prefix` produced a pkg-config file pointing at the wrong
  tree. It is now relocatable (`${pcfiledir}/../..`).
- `$<INSTALL_INTERFACE>` hard-coded `include` instead of
  `CMAKE_INSTALL_INCLUDEDIR`; install and export rules are now behind
  `MLRISK_INSTALL` so a parent project does not inherit them.
- A GCC `-Wmaybe-uninitialized` in the GARCH grid search would have failed
  the `-Werror` build on the Linux/gcc CI leg.
- `mlr_drawdown_scale` returned `MLR_EDOMAIN` for a non-finite equity value
  while every other function returns `MLR_EINVAL` for non-finite input; it
  now returns `MLR_EINVAL` (non-positive equity is still `MLR_EDOMAIN`).
- Header and README claims corrected: the PnL of a position is
  `position[t] * price[t-1] * returns[t]` (the price factor was missing);
  the GARCH constraint is `alpha + beta < 0.9999`; only alpha and beta are
  scale invariant; `n <= d` with `ridge == 0` returns `MLR_EDOMAIN` rather
  than fitting exactly; rolling std is O(n) on clean data with an O(window)
  rebuild after each gap.

### Added

- `tests/reference/reference_check.py`: 19 checks of the compiled C against
  pandas, numpy, scikit-learn, `arch`, exact rational arithmetic, an
  independent split generator, a bitwise no-lookahead sweep, a 400-fit
  GARCH Monte Carlo, and a reconciliation of the example's printed PnL.
  A `reference` CI job runs it on every push.
- Prefix-stability tests for EWMA and the rolling statistics (previously
  only the GARCH filter had one), tests for `window == n`, all-NaN rolling
  input, NULL outputs on the range estimators, and the GARCH filter
  overflow rule.
- `tests/test_fuzz.c`: 4000 randomized calls across the whole API (NaN, Inf,
  denormals, `SIZE_MAX` arguments) asserting the documented contracts; runs
  under the sanitizers in CI and found the two sizing/filter holes above.
- `tests/reference/real_data_check.py` for daily OHLC files, and
  `bench/bench.c` (`mlrisk_bench`) with per-element timings and scaling ratios.
- A 32-bit (`-m32`) CI leg, so the `size_t` guards are exercised where
  `size_t` is 32 bits; the workflow now runs on every branch push and can be
  dispatched by hand.
- The sanitizer CI job also runs the example; the Windows example step
  declares its shell; the workflow passes `actionlint`.
- The C two-pass reference in the rolling tests is computed on shifted
  values so that it is itself exact at large levels.


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
