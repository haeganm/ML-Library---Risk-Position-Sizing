# walkforward

> Volatility forecasts, position sizing and purged walk-forward splits that never see the future. A C11 library with a numpy front door.

```bash
pip install walkforward
```

Most backtests that look good are quietly peeking one bar ahead. The estimator that sizes a position knows the return it is about to earn, or a training label was computed from prices inside the test window. This library is built so those two things cannot happen by accident, and it is checked on every push against pandas, numpy, scikit-learn, the `arch` package and exact rational arithmetic.

- **One timing convention.** Every forecast at index `t` is built from data before `t`. `ewma_vol` and the GARCH filters are predictive; the rolling statistics and range estimators are contemporaneous and say so in their own docstrings, with `lag` to fix them.
- **Purging that asks the right question.** `PurgedWalkForward` takes a label horizon, not a purge count, because `purge = h - 1` is the part people get wrong. Drops straight into `cross_val_score`.
- **GARCH(1,1) by maximum likelihood**, checked against `arch` on identical samples and identical likelihoods, and invariant to the units of the returns.
- **Sizing that fails closed.** A bad price gives a zero position, not a NaN one. A leverage cap that could overflow is refused rather than silently ignored.
- **Ridge by Householder QR**, so a feature at a price level near 1e9 keeps its slope precision.

Research tooling, not investment advice. It is a set of building blocks, not a backtester: it knows nothing about costs, borrow, calendars or corporate actions.

## The loop

```python
import numpy as np
import walkforward as wf

# The fit assumes mean-zero returns. Demean with the TRAINING mean; the
# full-sample mean would put the future into the fit.
mu = returns[:1000].mean()
model = wf.garch_fit(returns[:1000] - mu)

# sigma[t] forecasts period t from returns before t. filter_from continues
# from the variance state the fit ended on, which is what makes the
# out-of-sample path the same as filtering everything together.
sigma = model.filter_from(returns[1000:] - mu)

# A position held over period t is entered at the close of t-1, so it is
# sized against the previous close, and earns position * price * return.
entry = close[999:-1]
position = wf.vol_target_position(
    sigma, target_vol=0.01, equity=100_000.0, price=entry, max_leverage=2.0
)
pnl = position * entry * returns[1000:]
```

`model.persistence`, `model.half_life` and `model.unconditional_vol` are there so you do not have to recompute them.

## Cross-validation

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import walkforward as wf

# Target is a 21-period forward return, so labels span 21 periods and the
# last 20 training rows before each test window are dropped.
cv = wf.PurgedWalkForward(train_size=756, test_size=252, label_horizon=21)

scores = cross_val_score(GradientBoostingRegressor(), X, y, cv=cv)
```

Without the purge, the twenty training rows before each test window carry labels that were partly computed from test-window prices. The library counts that leak in its own test suite: at a horizon of 21, a purge of 20 leaves zero leaking training rows and a purge of 19 leaves ten.

There is no `embargo` argument, on purpose. An embargo protects training data that sits after a test window, and a walk-forward never trains on anything after the window it is testing. `walk_forward_splits(..., include_post_train=True)` exposes that variant with the warning it deserves.

## A full example

[`examples/spy_walk_forward.ipynb`](https://github.com/haeganm/walkforward/blob/main/examples/spy_walk_forward.ipynb) runs one model end to end on 25 years of daily SPY: lagged features, a ridge on a five-day forward return, 41 purged folds, a GARCH per fold continued with `filter_from`, and 10% vol targeting sized against the previous close. Vol targeting alone lifts the out-of-sample Sharpe from 0.63 to 0.79 and halves the drawdown; the ridge signal has no edge and the notebook says so. The same model with the lags removed reports a Sharpe of 14.6, which is the leak this package exists to make hard.

## Conventions

Volatility is per period. Annualised converts as `annual / sqrt(periods_per_year)`, so a 10% annual target on daily data is `0.10 / sqrt(252)`, about 0.0063.

`rolling_std` uses the population convention and divides by `window`. Pandas `rolling().std()` defaults to the sample convention, so the two differ by `sqrt(window / (window - 1))`.

A pandas Series in gives a pandas Series out, on the same index. Realigning a bare array by hand is one of the ways lookahead gets in, so two Series passed together must share an index; `lag` is the way to shift one.

Bad arguments raise. Bad elements are handled per function and documented on each: a non-finite return is skipped by the recursions, makes a rolling window NaN, and gives a zero position in sizing. `DomainError` (a `ValueError`) means the arguments were fine but the computation has no answer: a singular design, a sample with no variance, an overflowing recursion.

## What it is checked against

The C library underneath is compared on every push to independent implementations. The numbers below come from `tests/reference/reference_check.py` in the repository, which needs only the packages it names.

| Check | Reference | Result |
|---|---|---|
| Rolling mean and std, with gaps | pandas | 3.1e-11 |
| Rolling std at a price level of 1e9 | exact rational arithmetic | 4.4e-16, where a two-pass computation is off by 4.9e-12 |
| EWMA, predictive alignment | pandas `ewm` shifted one period | 3.5e-18 |
| GARCH filter and forecast | `arch` | 4.4e-16 and 8.9e-16 relative |
| GARCH fit over 20 samples | `arch`, same likelihood | 5.0e-7 max parameter difference |
| Ridge, condition number 1e4 to 1e10 | SVD least squares | within 2x condition times epsilon |
| Ridge with a feature at level 1e6 to 1e15 | exact rational OLS | under 1e-14 |
| Walk-forward splits, 1620 parameter sets | independent generator | 0 mismatches |
| No lookahead, 40 trials | bitwise prefix comparison | 0 violations |

The C also runs under AddressSanitizer and UndefinedBehaviorSanitizer, on 32-bit and 64-bit, and gives bit-identical answers under GCC and Clang.

## License

MIT. Source and the C library: https://github.com/haeganm/walkforward
