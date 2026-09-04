// Randomized robustness sweep: every public function is called with random
// sizes and random contents, including NaN, Inf, denormals, huge values and
// negative zero. The contract checked is not the numbers (the reference
// suite does that) but that nothing crashes or reads out of bounds, every
// status is a known code, and the outputs that are promised finite are.
// Meant to run under AddressSanitizer and UBSan, which the CI sanitize job
// does.
#include "mlrisk/mlrisk.h"
#include "test_util.h"
#include <stdlib.h>
#include <string.h>

enum { MAX_N = 300, ITER = 4000 };

static double special_values[] = {0.0, -0.0, 1.0, -1.0, 1e-300, 5e-324, 1e300, -1e300,
                                  1e154, 1.7976931348623157e308, 0.01, -0.02};

static double random_value(unsigned long long *st) {
    double u = test_lcg_u01(st);
    if (u < 0.05) return MLR_NAN;
    if (u < 0.08) return INFINITY;
    if (u < 0.11) return -INFINITY;
    if (u < 0.30) return special_values[(size_t)(test_lcg_u01(st) * (sizeof special_values / sizeof special_values[0]))];
    return 0.02 * test_lcg_gauss(st);
}

static void fill(unsigned long long *st, double *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = random_value(st);
}

static int known_status(mlr_status s) {
    return s == MLR_OK || s == MLR_EINVAL || s == MLR_ENOMEM || s == MLR_EBOUNDS || s == MLR_EDOMAIN;
}

static int test_fuzz_sweep(void) {
    static double a[MAX_N], b[MAX_N], c[MAX_N], d[MAX_N], out[MAX_N];
    static mlr_split sp[MAX_N];
    unsigned long long st = 20260903ULL;

    for (int it = 0; it < ITER; it++) {
        size_t n = 1 + (size_t)(test_lcg_u01(&st) * MAX_N);
        fill(&st, a, n); fill(&st, b, n); fill(&st, c, n); fill(&st, d, n);
        double s1 = random_value(&st), s2 = random_value(&st), s3 = random_value(&st);
        size_t w = 1 + (size_t)(test_lcg_u01(&st) * (n + 2));
        mlr_status s;

        s = mlr_rolling_mean(a, n, w, out);
        ASSERT(known_status(s), "rolling_mean status");
        s = mlr_rolling_std(a, n, w, out);
        ASSERT(known_status(s), "rolling_std status");
        s = mlr_ewma_vol(a, n, s1, out);
        ASSERT(known_status(s), "ewma status");
        if (s == MLR_OK) {
            for (size_t i = 0; i < n; i++) {
                ASSERT(mlr_isnan(out[i]) || (mlr_isfinite(out[i]) && out[i] >= 0.0), "ewma output finite-or-NAN and >= 0");
            }
        }

        mlr_garch m;
        memset(&m, 0, sizeof m);
        s = mlr_garch_fit(a, n, &m);
        ASSERT(known_status(s), "garch_fit status");
        if (s == MLR_OK) {
            ASSERT(mlr_isfinite(m.omega) && m.omega > 0.0 && mlr_isfinite(m.loglik), "fit outputs finite");
            ASSERT(mlr_garch_filter(&m, b, n, out) == MLR_OK, "fitted model filters");
            for (size_t i = 0; i < n; i++) ASSERT(mlr_isfinite(out[i]) && out[i] > 0.0, "filter output finite and positive");
            ASSERT(mlr_garch_forecast(&m, 1 + (size_t)(test_lcg_u01(&st) * 50), out) == MLR_OK, "fitted model forecasts");
        }
        mlr_garch hand = {.omega = s1, .alpha = s2, .beta = s3, .sigma2_next = random_value(&st), .backcast = random_value(&st)};
        s = mlr_garch_filter(&hand, a, n, out);
        ASSERT(known_status(s), "hand-built filter status");
        if (s == MLR_OK) {
            for (size_t i = 0; i < n; i++) ASSERT(mlr_isfinite(out[i]) || mlr_isnan(out[i]), "hand-built filter output finite-or-NAN");
        }
        s = mlr_garch_forecast(&hand, 1 + (size_t)(test_lcg_u01(&st) * 50), out);
        ASSERT(known_status(s), "hand-built forecast status");

        s = mlr_parkinson_vol(a, b, n, out);
        ASSERT(known_status(s), "parkinson status");
        s = mlr_garman_klass_vol(a, b, c, d, n, out);
        ASSERT(known_status(s), "garman-klass status");
        if (s == MLR_OK) {
            for (size_t i = 0; i < n; i++) ASSERT(mlr_isnan(out[i]) || (mlr_isfinite(out[i]) && out[i] >= 0.0), "GK output finite-or-NAN and >= 0");
        }

        s = mlr_vol_target_position(a, s1, s2, b, s3, n, out);
        ASSERT(known_status(s), "vol_target status");
        if (s == MLR_OK) {
            for (size_t i = 0; i < n; i++) {
                ASSERT(mlr_isfinite(out[i]) && out[i] >= 0.0, "positions are finite and >= 0");
                ASSERT(!(out[i] * b[i] > s3 * s2 * (1.0 + 1e-12)), "notional never exceeds the cap");
            }
        }
        double f;
        s = mlr_kelly_fraction(a, n, s1, &f);
        ASSERT(known_status(s), "kelly status");
        if (s == MLR_OK) ASSERT(mlr_isfinite(f), "kelly output finite");
        s = mlr_drawdown_scale(a, n, s1, out);
        ASSERT(known_status(s), "drawdown status");
        if (s == MLR_OK) {
            for (size_t i = 0; i < n; i++) ASSERT(out[i] >= 0.0 && out[i] <= 1.0, "scale in [0, 1]");
        }

        size_t big = test_lcg_u01(&st) < 0.1 ? SIZE_MAX - (size_t)(test_lcg_u01(&st) * 10) : 0;
        size_t train = big ? big : 1 + (size_t)(test_lcg_u01(&st) * n);
        size_t test = 1 + (size_t)(test_lcg_u01(&st) * n);
        size_t step = test_lcg_u01(&st) < 0.1 ? SIZE_MAX : 1 + (size_t)(test_lcg_u01(&st) * n);
        size_t purge = (size_t)(test_lcg_u01(&st) * (train < n ? train : n));
        size_t embargo = test_lcg_u01(&st) < 0.2 ? SIZE_MAX - (size_t)(test_lcg_u01(&st) * 40) : (size_t)(test_lcg_u01(&st) * n);
        size_t count = 0;
        s = mlr_walk_forward_splits(n, train, test, step, purge, embargo, it & 1, sp, MAX_N, &count);
        ASSERT(known_status(s), "splits status");
        if (s == MLR_OK) {
            ASSERT(count <= MAX_N, "count within capacity");
            for (size_t i = 0; i < count; i++) {
                ASSERT(sp[i].train_start < sp[i].train_end && sp[i].train_end <= sp[i].test_start, "train before test");
                ASSERT(sp[i].test_start < sp[i].test_end && sp[i].test_end <= n, "test within bounds");
                ASSERT(sp[i].train_post_start >= sp[i].test_end && sp[i].train_post_end == n, "post segment after test");
            }
        }

        size_t dim = 1 + (size_t)(test_lcg_u01(&st) * 4);
        size_t rows = n / dim;
        if (rows > 0) {
            mlr_lin_model lm;
            ASSERT(mlr_lin_model_init(&lm, dim) == MLR_OK, "lin_model_init");
            s = mlr_linreg_fit(a, b, rows, dim, s1, &lm);
            ASSERT(known_status(s), "linreg_fit status");
            if (s == MLR_OK) {
                for (size_t j = 0; j < dim; j++) ASSERT(mlr_isfinite(lm.w[j]), "weights finite");
                ASSERT(mlr_isfinite(lm.b), "intercept finite");
                ASSERT(mlr_linreg_predict(c, rows, dim, &lm, out) == MLR_OK, "predict after fit");
            }
            mlr_lin_model_free(&lm);
        }
    }
    PASS("randomized sweep: no crash, known status codes, promised outputs finite");
}

int test_fuzz(void) {
    return test_fuzz_sweep();
}
