"""Tests for the Python layer.

The C library has its own suite and its own reference checks against pandas,
numpy, scikit-learn and arch. These tests cover what the binding adds: array
conversion, error mapping, pandas passthrough, the scikit-learn interface, and
that the timing guarantees survive the trip through Python.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import walkforward as wf

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def simulate(n=1500, omega=2e-6, alpha=0.10, beta=0.85, seed=0):
    """A GARCH(1,1) return series with a known data-generating process."""
    rng = np.random.default_rng(seed)
    s2 = omega / (1.0 - alpha - beta)
    out = np.empty(n)
    for t, z in enumerate(rng.standard_normal(n)):
        out[t] = math.sqrt(s2) * z
        s2 = omega + alpha * out[t] ** 2 + beta * s2
    return out


@pytest.fixture(scope="module")
def returns():
    return simulate()


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------


def test_package_and_library_versions_agree():
    assert wf.__version__ == wf.c_version()


def test_accepts_lists_and_non_contiguous_views():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    from_list = wf.rolling_mean(values, 2)
    strided = np.repeat(np.asarray(values), 2)[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(from_list, wf.rolling_mean(strided, 2))
    np.testing.assert_allclose(from_list[1:], [1.5, 2.5, 3.5, 4.5])


def test_accepts_integer_input_without_mutating_it():
    values = np.arange(5)
    result = wf.rolling_mean(values, 2)
    assert values.dtype == np.int64 or values.dtype == np.int32
    np.testing.assert_allclose(result[1:], [0.5, 1.5, 2.5, 3.5])


def test_read_only_input_is_accepted():
    values = np.arange(5.0)
    values.flags.writeable = False
    np.testing.assert_allclose(wf.rolling_mean(values, 2)[1:], [0.5, 1.5, 2.5, 3.5])


def test_output_never_aliases_input():
    values = np.arange(10.0)
    result = wf.rolling_mean(values, 3)
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize(
    "call, exception",
    [
        (lambda: wf.rolling_mean([1.0, 2.0], 0), ValueError),
        (lambda: wf.rolling_mean([], 1), ValueError),
        (lambda: wf.rolling_mean([[1.0, 2.0]], 1), ValueError),
        (lambda: wf.ewma_vol([0.01, 0.02], 1.0), ValueError),
        (lambda: wf.ewma_vol([0.01, 0.02], float("nan")), ValueError),
        (lambda: wf.rolling_mean([1.0, 2.0], -1), ValueError),
        (lambda: wf.rolling_mean([1.0, 2.0], 1.5), TypeError),
        (lambda: wf.kelly_fraction([0.01]), ValueError),
        (lambda: wf.kelly_fraction([0.01, 0.01]), wf.DomainError),
        (lambda: wf.garch_fit(np.zeros(200)), wf.DomainError),
        (lambda: wf.garch_fit(np.ones(10)), ValueError),
        (lambda: wf.vol_target_position([0.01], 0.01, float("nan"), [100.0], 2.0), ValueError),
        (lambda: wf.drawdown_scale([100.0, -1.0], 0.2), wf.DomainError),
        (lambda: wf.drawdown_scale([100.0, float("nan")], 0.2), ValueError),
    ],
)
def test_errors_map_to_python_exceptions(call, exception):
    with pytest.raises(exception):
        call()


def test_domain_error_is_a_value_error():
    assert issubclass(wf.DomainError, ValueError)


def test_length_mismatch_is_reported_clearly():
    with pytest.raises(ValueError, match="same length"):
        wf.parkinson_vol([1.0, 2.0, 3.0], [1.0, 2.0])


def test_masked_entries_are_missing_not_fill_values():
    masked = np.ma.array([1.0, 2.0, 999.0, 4.0], mask=[0, 0, 1, 0])
    with_nan = np.array([1.0, 2.0, np.nan, 4.0])
    np.testing.assert_array_equal(wf.rolling_mean(masked, 2), wf.rolling_mean(with_nan, 2))
    np.testing.assert_array_equal(wf.ewma_vol(masked), wf.ewma_vol(with_nan))
    assert math.isnan(wf.rolling_mean(masked, 4)[-1])


@pytest.mark.parametrize("value", [3.0, np.array(7.0), None])
def test_scalar_input_is_refused_not_promoted(value):
    with pytest.raises(ValueError, match="scalar"):
        wf.rolling_mean(value, 1)


def test_non_numeric_input_is_refused():
    with pytest.raises(TypeError, match="numeric"):
        wf.ewma_vol(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))


def test_object_arrays_with_none_still_read_as_missing():
    result = wf.rolling_mean(np.array([1.0, None, 3.0], dtype=object), 1)
    np.testing.assert_array_equal(result, [1.0, np.nan, 3.0])


def test_sparse_input_is_named_as_such():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match="sparse"):
        wf.rolling_mean(scipy_sparse.csr_matrix(np.eye(3)), 1)


def test_unreadable_elements_name_the_argument():
    pd = pytest.importorskip("pandas")
    with pytest.raises(TypeError, match="returns must be numeric"):
        wf.ewma_vol(pd.Series([0.1, pd.NA, 0.2], dtype=object))
    with pytest.raises(TypeError, match="x must be numeric"):
        wf.lag(np.array([1.0, "two"], dtype=object))


def test_domain_errors_say_why_not_what_was_valid():
    with pytest.raises(wf.DomainError, match="zero variance") as caught:
        wf.kelly_fraction([0.01, 0.01, 0.01])
    assert "must be finite" not in str(caught.value)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 2))
    X = np.column_stack([X, X[:, 0] + X[:, 1]])  # dependent column
    with pytest.raises(wf.DomainError, match="rank deficient") as caught:
        wf.Ridge().fit(X, rng.standard_normal(20))
    assert "must be finite" not in str(caught.value)
    with pytest.raises(wf.DomainError, match="more rows than columns"):
        wf.Ridge().fit(np.eye(3), np.ones(3))


def test_non_integer_counts_are_a_type_error():
    for bad in (float("inf"), float("nan"), 1.5, "3"):
        with pytest.raises(TypeError):
            wf.lag([1.0, 2.0], bad)


# --------------------------------------------------------------------------
# Timing guarantees
# --------------------------------------------------------------------------


def test_predictive_estimators_do_not_see_the_current_period(returns):
    """Rewriting everything after t must not move any output at or before t."""
    cut = 800
    changed = returns.copy()
    changed[cut + 1 :] = simulate(len(returns) - cut - 1, seed=99)

    model = wf.garch_fit(returns[:500])
    for original, altered in (
        (wf.ewma_vol(returns), wf.ewma_vol(changed)),
        (model.filter(returns), model.filter(changed)),
    ):
        np.testing.assert_array_equal(original[: cut + 1], altered[: cut + 1])


def test_range_estimators_are_contemporaneous():
    """out[t] moves when bar t moves, and only then."""
    rng = np.random.default_rng(3)
    n = 50
    low = 99 + rng.random(n)
    high = low + 1 + rng.random(n)
    base = wf.parkinson_vol(high, low)
    bumped_high = high.copy()
    bumped_high[20] += 0.5
    moved = np.flatnonzero(base != wf.parkinson_vol(bumped_high, low))
    assert moved.tolist() == [20]


def test_lag_makes_a_contemporaneous_estimator_safe():
    values = np.arange(5.0)
    lagged = wf.lag(wf.rolling_mean(values, 2))
    assert math.isnan(lagged[0]) and math.isnan(lagged[1])
    np.testing.assert_allclose(lagged[2:], [0.5, 1.5, 2.5])


def test_filter_from_continues_the_fit_exactly(returns):
    """The documented bit-exact continuation, through the binding."""
    model = wf.garch_fit(returns[:500])
    full = model.filter(returns)
    continued = model.filter_from(returns[500:])
    np.testing.assert_array_equal(continued, full[500:])
    # Filtering the tail alone restarts from the backcast and is not the same
    assert model.filter(returns[500:])[0] != full[500]


def test_hand_built_model_must_set_sigma2_next_before_forecasting():
    model = wf.GarchModel(omega=2e-6, alpha=0.1, beta=0.85)
    with pytest.raises(ValueError):
        model.forecast(5)
    with pytest.raises(ValueError):
        model.filter_from([0.01, 0.02])


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_garch_recovers_the_data_generating_process(returns):
    model = wf.garch_fit(returns)
    assert model.converged
    assert abs(model.alpha - 0.10) < 0.05
    assert abs(model.persistence - 0.95) < 0.05
    assert model.unconditional_vol > 0
    assert model.half_life > 0
    # sigma2_next is the state the in-sample path ends on
    path = model.filter(returns)
    expected = model.omega + model.alpha * returns[-1] ** 2 + model.beta * path[-1] ** 2
    assert abs(expected / model.sigma2_next - 1.0) < 1e-12


def test_garch_forecast_follows_its_decay_formula():
    model = wf.GarchModel(omega=1e-6, alpha=0.05, beta=0.90, sigma2_next=1e-4)
    assert model.persistence == pytest.approx(0.95)
    assert model.unconditional_variance == pytest.approx(1e-6 / 0.05)
    assert model.half_life == pytest.approx(math.log(0.5) / math.log(0.95))

    horizon = 500
    forecast = model.forecast(horizon)
    # sigma2[h] = uncond + persistence**h * (sigma2_next - uncond)
    gap = model.sigma2_next - model.unconditional_variance
    expected = np.sqrt(
        model.unconditional_variance + model.persistence ** np.arange(horizon) * gap
    )
    np.testing.assert_allclose(forecast, expected, rtol=1e-15)
    assert forecast[0] == pytest.approx(math.sqrt(model.sigma2_next), rel=1e-15)
    # Convergence is geometric, so the remaining gap is set by persistence**h
    assert forecast[-1] == pytest.approx(model.unconditional_vol, rel=1e-9)


def test_ewma_matches_its_recursion_definition():
    r = np.array([0.01, -0.02, 0.015, 0.03])
    lam = 0.9
    out = wf.ewma_vol(r, lam)
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(abs(r[0]))
    variance = r[0] ** 2
    variance = lam * variance + (1 - lam) * r[1] ** 2
    assert out[2] == pytest.approx(math.sqrt(variance))


def test_vol_targeting_hits_its_target(returns):
    """Sizing on a correct forecast should realise close to the target vol."""
    target = 0.01
    model = wf.garch_fit(returns[:750])
    sigma = model.filter_from(returns[750:])
    price = np.full(len(sigma), 100.0)
    position = wf.vol_target_position(sigma, target, 100_000.0, price, max_leverage=10.0)
    strategy = position * price * returns[750:] / 100_000.0
    assert 0.8 < strategy.std() / target < 1.25


def test_leverage_cap_binds():
    position = wf.vol_target_position([1e-6], 0.01, 10_000.0, [100.0], max_leverage=2.0)
    assert position[0] == pytest.approx(2.0 * 10_000.0 / 100.0)


def test_kelly_matches_mean_over_variance():
    rng = np.random.default_rng(7)
    r = 0.01 * rng.standard_normal(500) + 0.0005
    assert wf.kelly_fraction(r, 0.5) == pytest.approx(0.5 * r.mean() / r.var(ddof=1))


def test_drawdown_scale_tapers_to_zero():
    equity = np.array([100.0, 110.0, 99.0, 104.5, 120.0])
    scale = wf.drawdown_scale(equity, 0.2)
    np.testing.assert_allclose(scale, [1.0, 1.0, 0.5, 0.75, 1.0])


def test_ridge_matches_the_normal_equations_on_well_posed_data():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((200, 3))
    beta = np.array([1.5, -2.0, 0.25])
    y = X @ beta + 0.5 + 0.01 * rng.standard_normal(200)
    model = wf.Ridge().fit(X, y)
    np.testing.assert_allclose(model.coef_, beta, atol=2e-3)
    assert model.intercept_ == pytest.approx(0.5, abs=2e-3)
    np.testing.assert_allclose(model.predict(X[:5]), X[:5] @ model.coef_ + model.intercept_)


def test_ridge_refuses_a_singular_design():
    X = np.column_stack([np.arange(10.0), np.arange(10.0)])
    y = np.arange(10.0)
    with pytest.raises(wf.DomainError):
        wf.Ridge().fit(X, y)
    shrunk = wf.Ridge(ridge=1.0).fit(X, y)
    assert shrunk.coef_[0] == pytest.approx(shrunk.coef_[1])


def test_ridge_keeps_precision_at_a_large_feature_level():
    """A price-like column near 1e9 with unit variation."""
    rng = np.random.default_rng(5)
    x = 1e9 + rng.standard_normal(400)
    y = 2.0 * (x - 1e9) + 1.0
    model = wf.Ridge().fit(x.reshape(-1, 1), y)
    assert model.coef_[0] == pytest.approx(2.0, rel=1e-9)


def test_ridge_is_unfitted_until_fit_succeeds():
    model = wf.Ridge()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((2, 1)))


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_splits_are_ordered_and_purged():
    splits = wf.walk_forward_splits(1000, train_size=200, test_size=100, label_horizon=21)
    assert splits
    for split in splits:
        assert split.train_start < split.train_end <= split.test_start < split.test_end
        assert split.purge == 20
        assert split.test_end <= 1000


def test_purge_is_exactly_the_leakage_boundary():
    """A label spanning h periods must not reach into the test window."""

    def leaking(horizon, splits):
        return sum(
            1
            for split in splits
            for i in range(split.train_start, split.train_end)
            if any(split.test_start <= p < split.test_end for p in range(i, i + horizon))
        )

    for horizon in (2, 5, 21):
        clean = wf.walk_forward_splits(400, 100, 30, label_horizon=horizon)
        under = wf.walk_forward_splits(400, 100, 30, purge=horizon - 2)
        assert leaking(horizon, clean) == 0
        assert leaking(horizon, under) > 0


def test_split_slices_index_the_data():
    x = np.arange(500.0)
    split = wf.walk_forward_splits(500, train_size=100, test_size=50)[0]
    assert x[split.train].shape == (100,)
    assert x[split.test].shape == (50,)
    assert x[split.train_post].size == 0


def test_too_little_data_yields_no_splits():
    assert wf.walk_forward_splits(10, train_size=100, test_size=50) == []


def test_purge_larger_than_the_training_window_is_rejected():
    with pytest.raises(ValueError, match="label horizon of 20"):
        wf.walk_forward_splits(500, train_size=10, test_size=5, label_horizon=20)
    with pytest.raises(ValueError) as caught:
        wf.walk_forward_splits(500, train_size=10, test_size=5, purge=20)
    assert "horizon" not in str(caught.value)


def test_post_train_segment_is_opt_in():
    without = wf.walk_forward_splits(500, 100, 50)[0]
    with_post = wf.walk_forward_splits(500, 100, 50, embargo=10, include_post_train=True)[0]
    assert without.train_post_start == without.train_post_end
    assert with_post.train_post_start == with_post.test_end + 10


def test_cross_validator_interface():
    X = np.zeros((1000, 2))
    cv = wf.PurgedWalkForward(train_size=200, test_size=100, label_horizon=21)
    assert cv.get_n_splits(X) == len(list(cv.split(X)))
    for train_idx, test_idx in cv.split(X):
        assert train_idx.max() + 20 < test_idx.min()
        assert np.intersect1d(train_idx, test_idx).size == 0
    assert "label_horizon=21" in repr(cv)


def test_test_windows_tile_the_sample_by_default():
    cv = wf.PurgedWalkForward(train_size=200, test_size=100)
    tested = np.concatenate([test for _, test in cv.split(np.zeros(1000))])
    assert np.array_equal(tested, np.arange(200, 1000))
    # A tail shorter than one test window is not tested, as documented.
    tested = np.concatenate([test for _, test in cv.split(np.zeros(1050))])
    assert np.array_equal(tested, np.arange(200, 1000))


# --------------------------------------------------------------------------
# Optional integrations
# --------------------------------------------------------------------------


def test_pandas_input_keeps_its_index():
    pd = pytest.importorskip("pandas")
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    series = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], index=index, name="ret")
    result = wf.ewma_vol(series, 0.94)
    assert isinstance(result, pd.Series)
    assert result.index.equals(index)
    assert result.name == "ret"
    np.testing.assert_array_equal(result.to_numpy(), wf.ewma_vol(series.to_numpy(), 0.94))


def test_paired_series_must_share_an_index():
    pd = pytest.importorskip("pandas")
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    high = pd.Series([101.0, 102.0, 103.0, 104.0, 105.0], index=index)
    low = pd.Series([99.0, 100.0, 101.0, 102.0, 103.0], index=index)
    aligned = wf.parkinson_vol(high, low)
    np.testing.assert_array_equal(aligned.to_numpy(), wf.parkinson_vol(high.to_numpy(), low.to_numpy()))
    with pytest.raises(ValueError, match="different indexes"):
        wf.parkinson_vol(high, low.sort_index(ascending=False))
    with pytest.raises(ValueError, match="different indexes"):
        wf.vol_target_position(high[1:], 0.01, 1.0, high[:-1], 2.0)
    # Realigning with lag keeps the index and is the documented way.
    sized = wf.vol_target_position(wf.ewma_vol(high.pct_change().fillna(0.0)), 0.01, 1.0, wf.lag(high), 2.0)
    assert sized.index.equals(index)


def test_works_with_sklearn_cross_val_score():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    del sklearn
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 3))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * rng.standard_normal(600)

    cv = wf.PurgedWalkForward(train_size=200, test_size=100, label_horizon=5)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv)
    assert len(scores) == cv.get_n_splits(X)
    assert scores.mean() > 0.9


def test_ridge_refuses_reordered_dataframe_columns():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    y = frame @ np.array([1.0, 2.0, 3.0])
    model = wf.Ridge().fit(frame, y)
    assert model.n_features_in_ == 3
    assert list(model.feature_names_in_) == ["a", "b", "c"]
    np.testing.assert_allclose(model.predict(frame), y, atol=1e-10)
    with pytest.raises(ValueError, match="same columns in the same order"):
        model.predict(frame[["c", "b", "a"]])
    # A plain array carries no names and is accepted positionally, as before
    np.testing.assert_allclose(model.predict(frame.to_numpy()), y, atol=1e-10)
    # Refitting on an array forgets the names
    model.fit(frame.to_numpy(), y)
    assert not hasattr(model, "feature_names_in_")


def test_ridge_unfitted_predict_is_a_not_fitted_error():
    sklearn_exceptions = pytest.importorskip("sklearn.exceptions")
    with pytest.raises(sklearn_exceptions.NotFittedError):
        wf.Ridge().predict(np.zeros((2, 2)))


def test_ridge_argument_messages():
    with pytest.raises(TypeError, match="ridge must be a number"):
        wf.Ridge(ridge="1.0").fit(np.eye(3), np.ones(3))
    with pytest.raises(ValueError, match="no features"):
        wf.Ridge().fit(np.zeros((3, 0)), np.ones(3))
    with pytest.raises(ValueError, match="needs y"):
        wf.Ridge().fit(np.eye(3), None)


def test_huge_split_counts_are_refused_before_allocation():
    with pytest.raises(ValueError, match="splits"):
        wf.walk_forward_splits(2**62, train_size=1, test_size=1)
    # and the count itself is arithmetic, not a loop over 2**62 windows

    class Huge:
        def __len__(self):
            return 2**62

    assert wf.PurgedWalkForward(train_size=1, test_size=1).get_n_splits(Huge()) == 2**62 - 1


def test_ridge_is_a_usable_sklearn_estimator():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = wf.Ridge(ridge=0.5)
    assert clone(model).get_params() == {"ridge": 0.5}
