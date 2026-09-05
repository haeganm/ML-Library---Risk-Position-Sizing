#include "mlrisk/rolling.h"
#include "test_util.h"

#define TOL 1e-9

static int test_rolling_mean_basic(void) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5];

    ASSERT(mlr_rolling_mean(x, 5, 3, out) == MLR_OK, "rolling_mean returns MLR_OK");
    ASSERT(mlr_isnan(out[0]) && mlr_isnan(out[1]), "warmup is NAN");
    ASSERT_NEAR(out[2], 2.0, TOL, "mean of [1,2,3]");
    ASSERT_NEAR(out[3], 3.0, TOL, "mean of [2,3,4]");
    ASSERT_NEAR(out[4], 4.0, TOL, "mean of [3,4,5]");
    PASS("rolling_mean basic");
}

static int test_rolling_mean_edge_cases(void) {
    double x[] = {1.0};
    double out[1];

    ASSERT(mlr_rolling_mean(x, 1, 1, out) == MLR_OK, "window=1 OK");
    ASSERT_NEAR(out[0], 1.0, TOL, "window=1 mean is the value");

    ASSERT(mlr_rolling_mean(x, 1, 5, out) == MLR_OK, "window>n OK");
    ASSERT(mlr_isnan(out[0]), "window>n gives NAN");

    ASSERT(mlr_rolling_mean(NULL, 1, 1, out) == MLR_EINVAL, "NULL x -> EINVAL");
    ASSERT(mlr_rolling_mean(x, 1, 1, NULL) == MLR_EINVAL, "NULL out -> EINVAL");
    ASSERT(mlr_rolling_mean(x, 0, 1, out) == MLR_EINVAL, "n=0 -> EINVAL");
    ASSERT(mlr_rolling_mean(x, 1, 0, out) == MLR_EINVAL, "window=0 -> EINVAL");
    PASS("rolling_mean edge cases");
}

static int test_rolling_std_basic(void) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5];

    ASSERT(mlr_rolling_std(x, 5, 3, out) == MLR_OK, "rolling_std returns MLR_OK");
    ASSERT(mlr_isnan(out[0]) && mlr_isnan(out[1]), "warmup is NAN");
    // population variance of [1,2,3] = 2/3
    ASSERT_NEAR(out[2], sqrt(2.0 / 3.0), TOL, "std of [1,2,3]");
    ASSERT_NEAR(out[4], sqrt(2.0 / 3.0), TOL, "std of [3,4,5]");
    PASS("rolling_std basic");
}

// Two-pass population statistics over x[lo..hi], the reference the O(n)
// algorithms are checked against. Computed on values shifted by x[lo] so the
// reference itself stays exact at large levels (a plain two-pass at 1e9
// rounds its mean at 1e-5 and its std at 1e-11).
static void naive_stats(const double *x, size_t lo, size_t hi, double *mean_out, double *std_out) {
    double base = x[lo];
    double mean = 0.0;
    for (size_t j = lo; j <= hi; j++) mean += x[j] - base;
    mean /= (double)(hi - lo + 1);
    double var = 0.0;
    for (size_t j = lo; j <= hi; j++) {
        double d = (x[j] - base) - mean;
        var += d * d;
    }
    var /= (double)(hi - lo + 1);
    *mean_out = base + mean;
    *std_out = sqrt(var);
}

static int test_rolling_sliding_consistency(void) {
    enum { N = 1000 };
    static double x[N], fast_mean[N], fast_std[N];
    unsigned long long state = 42;
    for (size_t i = 0; i < N; i++) {
        x[i] = (test_lcg_u01(&state) - 0.5) * 0.05;
    }

    size_t windows[] = {2, 5, 50};
    for (size_t wi = 0; wi < 3; wi++) {
        size_t w = windows[wi];
        ASSERT(mlr_rolling_mean(x, N, w, fast_mean) == MLR_OK, "sliding mean OK");
        ASSERT(mlr_rolling_std(x, N, w, fast_std) == MLR_OK, "sliding std OK");
        for (size_t i = w - 1; i < N; i++) {
            double mean, sd;
            naive_stats(x, i - w + 1, i, &mean, &sd);
            ASSERT_NEAR(fast_mean[i], mean, TOL, "sliding mean matches two-pass");
            ASSERT_NEAR(fast_std[i], sd, TOL, "sliding std matches two-pass");
        }
    }
    PASS("sliding algorithms match two-pass recompute");
}

static int test_rolling_std_edge_cases(void) {
    double constant[10] = {3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5};
    double out[10];

    ASSERT(mlr_rolling_std(constant, 10, 4, out) == MLR_OK, "constant std OK");
    for (size_t i = 3; i < 10; i++) {
        ASSERT(out[i] == 0.0, "std of a constant is exactly 0");
    }

    ASSERT(mlr_rolling_std(constant, 3, 5, out) == MLR_OK, "window>n std OK");
    for (size_t i = 0; i < 3; i++) {
        ASSERT(mlr_isnan(out[i]), "window>n std is NAN");
    }

    ASSERT(mlr_rolling_std(constant, 3, 1, out) == MLR_OK, "window=1 std OK");
    ASSERT(out[0] == 0.0 && out[2] == 0.0, "window=1 std is 0");

    // window == n: a single full window at the last index
    double x[] = {1.0, 2.0, 3.0, 4.0};
    ASSERT(mlr_rolling_mean(x, 4, 4, out) == MLR_OK, "window=n mean OK");
    ASSERT(mlr_isnan(out[2]) && out[3] == 2.5, "window=n mean is the full-sample mean");
    ASSERT(mlr_rolling_std(x, 4, 4, out) == MLR_OK, "window=n std OK");
    ASSERT_NEAR(out[3], sqrt(1.25), 1e-12, "window=n std is the full-sample std");

    // All non-finite input: every output NAN, no error
    double bad[] = {MLR_NAN, INFINITY, MLR_NAN};
    ASSERT(mlr_rolling_mean(bad, 3, 2, out) == MLR_OK, "all-NAN mean OK");
    ASSERT(mlr_isnan(out[1]) && mlr_isnan(out[2]), "all-NAN mean gives NAN");
    ASSERT(mlr_rolling_std(bad, 3, 2, out) == MLR_OK, "all-NAN std OK");
    ASSERT(mlr_isnan(out[1]) && mlr_isnan(out[2]), "all-NAN std gives NAN");

    ASSERT(mlr_rolling_std(x, 0, 1, out) == MLR_EINVAL, "std n=0 -> EINVAL");
    ASSERT(mlr_rolling_std(NULL, 1, 1, out) == MLR_EINVAL, "std NULL x -> EINVAL");
    ASSERT(mlr_rolling_std(x, 1, 1, NULL) == MLR_EINVAL, "std NULL out -> EINVAL");
    PASS("rolling_std edge cases");
}

static int test_rolling_shift_invariance(void) {
    // A series at level 1e9 with unit-scale variation. The two-pass
    // reference on the same data isolates the algorithm from the input
    // representation, so the sliding result must match it tightly; without
    // offset shifting the rolling Welford removal step loses ~6 digits here.
    enum { N = 500, W = 20 };
    static double shifted[N], std_out[N], mean_out[N];
    const double OFFSET = 1e9;

    unsigned long long state = 777;
    for (size_t i = 0; i < N; i++) {
        shifted[i] = OFFSET + (test_lcg_u01(&state) - 0.5);
    }

    ASSERT(mlr_rolling_std(shifted, N, W, std_out) == MLR_OK, "shifted std OK");
    ASSERT(mlr_rolling_mean(shifted, N, W, mean_out) == MLR_OK, "shifted mean OK");

    for (size_t i = W - 1; i < N; i++) {
        double mean, sd;
        naive_stats(shifted, i - W + 1, i, &mean, &sd);
        ASSERT_NEAR(std_out[i], sd, 1e-12, "std at level 1e9 matches the exact reference to 1e-12");
        ASSERT_NEAR(mean_out[i], mean, 1e-6, "mean at level 1e9 matches the exact reference");
    }
    PASS("shift invariance at level 1e9");
}

static int test_rolling_outliers_and_trend(void) {
    // The sliding accumulators must not remember an outlier after it has
    // left the window, and must not lose precision as the level drifts away
    // from where the series started. Reference: two-pass on values shifted
    // by the window's own first element, which is exact to rounding.
    enum { N = 400, W = 20 };
    static double x[N], mean_out[N], std_out[N];
    unsigned long long state = 2718;

    double firsts[] = {1e9, 1e12, 1e15};
    for (size_t f = 0; f < 3; f++) {
        for (size_t i = 0; i < N; i++) x[i] = 100.0 + (test_lcg_u01(&state) - 0.5);
        x[0] = firsts[f];
        x[200] = -firsts[f];
        ASSERT(mlr_rolling_mean(x, N, W, mean_out) == MLR_OK, "mean with outliers OK");
        ASSERT(mlr_rolling_std(x, N, W, std_out) == MLR_OK, "std with outliers OK");
        for (size_t i = W - 1; i < N; i++) {
            double mean, sd;
            naive_stats(x, i - W + 1, i, &mean, &sd);
            int contains_outlier = (i - W + 1 == 0) || (i >= 200 && i - W + 1 <= 200);
            if (contains_outlier) continue;
            ASSERT_NEAR(mean_out[i], mean, 1e-12, "mean of a clean window is unaffected by an outlier that left");
            ASSERT_NEAR(std_out[i], sd, 1e-12, "std of a clean window is unaffected by an outlier that left");
        }
    }

    // Trend from 100 to 1e6 with unit noise
    enum { T = 20000 };
    static double trend[T], tm[T], ts[T];
    for (size_t i = 0; i < T; i++) trend[i] = 100.0 + (1e6 - 100.0) * (double)i / (T - 1) + (test_lcg_u01(&state) - 0.5);
    ASSERT(mlr_rolling_mean(trend, T, 50, tm) == MLR_OK, "trend mean OK");
    ASSERT(mlr_rolling_std(trend, T, 50, ts) == MLR_OK, "trend std OK");
    for (size_t i = 49; i < T; i += 97) {
        double mean, sd;
        naive_stats(trend, i - 49, i, &mean, &sd);
        ASSERT_NEAR(tm[i] / mean, 1.0, 1e-14, "mean along a long trend");
        ASSERT_NEAR(ts[i] / sd, 1.0, 1e-12, "std along a long trend");
    }
    PASS("outliers leaving the window and long trends");
}

static int test_rolling_std_spike_just_missing_the_rebuild_guard(void) {
    // A spike that takes most of the variance with it when it leaves, but
    // not enough to trip a near-total-collapse guard, leaves cancellation
    // error that compounds until the next periodic rebuild. Spike sizes are
    // chosen so the variance drops by 1e-2 to 1e-6 on exit, the range a
    // guard at 1e-6 let through.
    enum { N = 400, W = 20 };
    static double x[N], std_out[N];
    unsigned long long state = 31415;
    double spikes[] = {30.0, 300.0, 900.0};
    for (size_t k = 0; k < 3; k++) {
        for (size_t i = 0; i < N; i++) x[i] = 100.0 + (test_lcg_u01(&state) - 0.5);
        x[60] += spikes[k];
        x[61] -= spikes[k] * 0.3;  // a second, smaller collapse two steps later
        ASSERT(mlr_rolling_std(x, N, W, std_out) == MLR_OK, "std with a spike OK");
        for (size_t i = W - 1; i < N; i++) {
            if (i - W + 1 <= 61 && i >= 60) continue;  // window still holds a spike
            double mean, sd;
            naive_stats(x, i - W + 1, i, &mean, &sd);
            ASSERT_NEAR(std_out[i] / sd, 1.0, 1e-13, "std after a spike that just missed the guard");
        }
    }
    PASS("spike leaving the window below the old rebuild threshold");
}

static int test_rolling_overflow_windows(void) {
    // Finite inputs whose differences overflow: the window is NaN, never
    // Inf, never a silent zero std, and the state recovers afterwards
    double x[] = {1e308, -1e308, 1e308, -1e308, 1.0, 2.0, 3.0, 4.0};
    double mean_out[8], std_out[8];
    ASSERT(mlr_rolling_mean(x, 8, 2, mean_out) == MLR_OK, "mean OK");
    ASSERT(mlr_rolling_std(x, 8, 2, std_out) == MLR_OK, "std OK");
    for (size_t i = 1; i < 4; i++) {
        ASSERT(mlr_isnan(mean_out[i]), "mean of an overflowing window is NaN");
        ASSERT(mlr_isnan(std_out[i]), "std of an overflowing window is NaN, not 0");
    }
    ASSERT(mean_out[4] == -5e307, "representable mean of {-1e308, 1} is computed");
    ASSERT(mlr_isnan(std_out[4]), "std whose variance overflows is NaN");
    ASSERT(mean_out[5] == 1.5 && mean_out[7] == 3.5, "mean recovers after the overflow leaves");
    ASSERT(std_out[5] == 0.5 && std_out[7] == 0.5, "std recovers after the overflow leaves");
    for (size_t i = 0; i < 8; i++) {
        ASSERT(!(mlr_isfinite(mean_out[i]) == 0 && !mlr_isnan(mean_out[i])), "no Inf in mean output");
        ASSERT(!(mlr_isfinite(std_out[i]) == 0 && !mlr_isnan(std_out[i])), "no Inf in std output");
    }
    PASS("overflowing windows are NaN and recover");
}

static int test_rolling_nan_recovery(void) {
    // A bad value affects only the windows containing it
    enum { N = 20, W = 3 };
    double x[N], mean_out[N], std_out[N];
    unsigned long long state = 4242;
    for (size_t i = 0; i < N; i++) {
        x[i] = test_lcg_u01(&state) - 0.5;
    }
    x[7] = MLR_NAN;
    x[13] = INFINITY;

    ASSERT(mlr_rolling_mean(x, N, W, mean_out) == MLR_OK, "mean with NaN OK");
    ASSERT(mlr_rolling_std(x, N, W, std_out) == MLR_OK, "std with NaN OK");

    for (size_t i = W - 1; i < N; i++) {
        int has_bad = 0;
        for (size_t j = i - W + 1; j <= i; j++) {
            if (!mlr_isfinite(x[j])) has_bad = 1;
        }
        if (has_bad) {
            ASSERT(mlr_isnan(mean_out[i]), "window with bad value gives NAN mean");
            ASSERT(mlr_isnan(std_out[i]), "window with bad value gives NAN std");
        } else {
            double mean, sd;
            naive_stats(x, i - W + 1, i, &mean, &sd);
            ASSERT_NEAR(mean_out[i], mean, TOL, "mean recovers after bad value");
            ASSERT_NEAR(std_out[i], sd, TOL, "std recovers after bad value");
        }
    }

    ASSERT(mlr_rolling_std(x, N, 1, std_out) == MLR_OK, "window=1 with NaN OK");
    ASSERT(mlr_isnan(std_out[7]), "window=1 std of NaN is NAN");
    ASSERT(std_out[8] == 0.0, "window=1 std of a finite value is 0");
    PASS("NaN recovery");
}

static int test_ewma_vol_known_answer(void) {
    // Predictive: out[t] uses returns[0..t-1]
    double returns[] = {0.01, -0.02, 0.015};
    double out[3];
    double lambda = 0.9;

    ASSERT(mlr_ewma_vol(returns, 3, lambda, out) == MLR_OK, "ewma returns MLR_OK");
    ASSERT(mlr_isnan(out[0]), "no forecast before the first return");
    ASSERT_NEAR(out[1], 0.01, TOL, "out[1] = |r[0]|");
    double var = lambda * 0.01 * 0.01 + (1.0 - lambda) * 0.02 * 0.02;
    ASSERT_NEAR(out[2], sqrt(var), TOL, "out[2] from r[0], r[1] only");

    // lambda = 0: out[t] = |r[t-1]|
    ASSERT(mlr_ewma_vol(returns, 3, 0.0, out) == MLR_OK, "lambda=0 OK");
    ASSERT_NEAR(out[2], 0.02, TOL, "lambda=0 gives |r[t-1]|");
    PASS("ewma_vol known answer");
}

static int test_ewma_vol_invalid_inputs(void) {
    double returns[] = {0.01};
    double out[1];

    ASSERT(mlr_ewma_vol(returns, 1, -0.1, out) == MLR_EINVAL, "negative lambda -> EINVAL");
    ASSERT(mlr_ewma_vol(returns, 1, 1.0, out) == MLR_EINVAL, "lambda=1 (frozen) -> EINVAL");
    ASSERT(mlr_ewma_vol(returns, 1, 1.5, out) == MLR_EINVAL, "lambda>1 -> EINVAL");
    ASSERT(mlr_ewma_vol(returns, 1, MLR_NAN, out) == MLR_EINVAL, "NAN lambda -> EINVAL");
    ASSERT(mlr_ewma_vol(NULL, 1, 0.9, out) == MLR_EINVAL, "NULL returns -> EINVAL");
    ASSERT(mlr_ewma_vol(returns, 1, 0.9, NULL) == MLR_EINVAL, "NULL out -> EINVAL");
    ASSERT(mlr_ewma_vol(returns, 0, 0.9, out) == MLR_EINVAL, "n=0 -> EINVAL");
    PASS("ewma_vol invalid inputs");
}

static int test_ewma_vol_missing_data(void) {
    // Leading NANs are skipped, the first finite return seeds the variance,
    // an interior NAN neither changes the forecast already made nor the state
    double returns[] = {MLR_NAN, 0.02, MLR_NAN, 0.01, 0.03};
    double out[5];
    double lambda = 0.9;

    ASSERT(mlr_ewma_vol(returns, 5, lambda, out) == MLR_OK, "ewma with NANs OK");
    ASSERT(mlr_isnan(out[0]) && mlr_isnan(out[1]), "no forecast until a return is seen");
    ASSERT_NEAR(out[2], 0.02, TOL, "forecast after the seed is |r[1]|");
    ASSERT_NEAR(out[3], 0.02, TOL, "missing r[2] leaves the state unchanged");
    double var = lambda * 0.02 * 0.02 + (1.0 - lambda) * 0.01 * 0.01;
    ASSERT_NEAR(out[4], sqrt(var), TOL, "recursion resumes after the gap");

    // A return whose square overflows is treated as missing
    double huge[] = {0.01, 1e200, 0.01};
    ASSERT(mlr_ewma_vol(huge, 3, lambda, out) == MLR_OK, "overflowing return OK");
    ASSERT_NEAR(out[2], 0.01, TOL, "overflowing return does not poison the state");

    double all_nan[] = {MLR_NAN, MLR_NAN};
    ASSERT(mlr_ewma_vol(all_nan, 2, lambda, out) == MLR_OK, "all-NAN input OK");
    ASSERT(mlr_isnan(out[0]) && mlr_isnan(out[1]), "all-NAN input gives all NAN");
    PASS("ewma_vol missing data");
}

static int test_rolling_prefix_stability(void) {
    // Output at t depends only on x[0..t]: recomputing on a prefix gives the
    // prefix of the full result, bit for bit
    enum { N = 300, HALF = 150 };
    static double x[N], full[N], prefix[HALF];
    unsigned long long state = 31;
    for (size_t i = 0; i < N; i++) x[i] = 1e4 + (test_lcg_u01(&state) - 0.5);
    x[40] = MLR_NAN;

    size_t windows[] = {1, 7, 50};
    for (size_t wi = 0; wi < 3; wi++) {
        ASSERT(mlr_rolling_mean(x, N, windows[wi], full) == MLR_OK, "mean full OK");
        ASSERT(mlr_rolling_mean(x, HALF, windows[wi], prefix) == MLR_OK, "mean prefix OK");
        for (size_t i = 0; i < HALF; i++) {
            ASSERT(full[i] == prefix[i] || (mlr_isnan(full[i]) && mlr_isnan(prefix[i])),
                   "rolling mean is prefix-stable");
        }
        ASSERT(mlr_rolling_std(x, N, windows[wi], full) == MLR_OK, "std full OK");
        ASSERT(mlr_rolling_std(x, HALF, windows[wi], prefix) == MLR_OK, "std prefix OK");
        for (size_t i = 0; i < HALF; i++) {
            ASSERT(full[i] == prefix[i] || (mlr_isnan(full[i]) && mlr_isnan(prefix[i])),
                   "rolling std is prefix-stable");
        }
    }

    ASSERT(mlr_ewma_vol(x, N, 0.94, full) == MLR_OK, "ewma full OK");
    ASSERT(mlr_ewma_vol(x, HALF, 0.94, prefix) == MLR_OK, "ewma prefix OK");
    for (size_t i = 0; i < HALF; i++) {
        ASSERT(full[i] == prefix[i] || (mlr_isnan(full[i]) && mlr_isnan(prefix[i])),
               "ewma is prefix-stable");
    }
    PASS("prefix stability");
}

int test_rolling(void) {
    int failures = 0;
    failures += test_rolling_mean_basic();
    failures += test_rolling_mean_edge_cases();
    failures += test_rolling_std_basic();
    failures += test_rolling_sliding_consistency();
    failures += test_rolling_std_edge_cases();
    failures += test_rolling_shift_invariance();
    failures += test_rolling_nan_recovery();
    failures += test_rolling_outliers_and_trend();
    failures += test_rolling_std_spike_just_missing_the_rebuild_guard();
    failures += test_rolling_overflow_windows();
    failures += test_ewma_vol_known_answer();
    failures += test_ewma_vol_invalid_inputs();
    failures += test_ewma_vol_missing_data();
    failures += test_rolling_prefix_stability();
    return failures;
}
