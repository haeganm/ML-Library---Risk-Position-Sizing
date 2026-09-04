#include "mlrisk/vol.h"
#include "mlrisk/rolling.h"
#include "test_util.h"

#define TOL 1e-9

static int test_parkinson_known_answer(void) {
    // high = 100*e^0.02, low = 100 -> sigma = 0.02 / sqrt(4 ln 2)
    double high[] = {100.0 * exp(0.02), 100.0, 99.0, 1e308};
    double low[] = {100.0, 100.0, 100.0, 1e-308};
    double out[4];

    ASSERT(mlr_parkinson_vol(high, low, 4, out) == MLR_OK, "parkinson returns MLR_OK");
    ASSERT_NEAR(out[0], 0.02 / sqrt(4.0 * log(2.0)), TOL, "parkinson known answer");
    ASSERT(out[1] == 0.0, "zero range gives zero vol");
    ASSERT(mlr_isnan(out[2]), "high < low gives NAN");
    ASSERT(mlr_isnan(out[3]), "overflowing ratio gives NAN");

    ASSERT(mlr_parkinson_vol(NULL, low, 3, out) == MLR_EINVAL, "NULL high -> EINVAL");
    ASSERT(mlr_parkinson_vol(high, low, 3, NULL) == MLR_EINVAL, "NULL out -> EINVAL");
    ASSERT(mlr_parkinson_vol(high, low, 0, out) == MLR_EINVAL, "n=0 -> EINVAL");
    PASS("parkinson known answer");
}

static int test_garman_klass_known_answer(void) {
    // Bar 0: open == close kills the second term -> sigma = sqrt(0.5)*ln(h/l)
    // Bar 1: close above high (inconsistent bar) -> NAN
    // Bar 2: zero close -> NAN
    // Bar 3: open below low -> NAN
    // Bar 4: open at the low, close at the high: the smallest possible
    //        estimate for the range, (0.5 - (2 ln 2 - 1)) * hl^2, still positive
    double open[] = {100.0, 100.0, 100.0, 99.0, 100.0};
    double high[] = {102.0, 100.5, 102.0, 102.0, 101.0};
    double low[] = {100.0, 99.9, 100.0, 100.0, 100.0};
    double close[] = {100.0, 110.0, 0.0, 101.0, 101.0};
    double out[5];

    ASSERT(mlr_garman_klass_vol(open, high, low, close, 5, out) == MLR_OK, "garman-klass returns MLR_OK");
    ASSERT_NEAR(out[0], sqrt(0.5) * log(102.0 / 100.0), TOL, "garman-klass known answer");
    ASSERT(mlr_isnan(out[1]), "close outside [low, high] gives NAN");
    ASSERT(mlr_isnan(out[2]), "non-positive close gives NAN");
    ASSERT(mlr_isnan(out[3]), "open outside [low, high] gives NAN");
    double hl = log(1.01);
    ASSERT_NEAR(out[4], sqrt((0.5 - (2.0 * log(2.0) - 1.0)) * hl * hl), TOL,
                "full-range move gives the minimum positive estimate");

    ASSERT(mlr_garman_klass_vol(NULL, high, low, close, 3, out) == MLR_EINVAL, "NULL open -> EINVAL");
    ASSERT(mlr_garman_klass_vol(open, high, low, close, 3, NULL) == MLR_EINVAL, "NULL out -> EINVAL");
    ASSERT(mlr_garman_klass_vol(open, high, low, close, 0, out) == MLR_EINVAL, "n=0 -> EINVAL");
    PASS("garman-klass known answer");
}

static int test_garch_filter_known_answer(void) {
    // backcast = 0 selects the unconditional variance omega/(1-alpha-beta) = 1,
    // a fixed point of the recursion, so sigma2[0] = 1
    mlr_garch model = {.omega = 0.2, .alpha = 0.3, .beta = 0.5, .converged = 1, .backcast = 0.0};
    double returns[] = {1.0, -2.0, 3.0};
    double sigma[3];

    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_OK, "garch_filter returns MLR_OK");
    double s2_0 = 1.0;
    double s2_1 = 0.2 + 0.3 * 1.0 + 0.5 * s2_0;
    double s2_2 = 0.2 + 0.3 * 4.0 + 0.5 * s2_1;
    ASSERT_NEAR(sigma[0], sqrt(s2_0), 1e-12, "filter step 0 (unconditional seed)");
    ASSERT_NEAR(sigma[1], sqrt(s2_1), 1e-12, "filter step 1");
    ASSERT_NEAR(sigma[2], sqrt(s2_2), 1e-12, "filter step 2");

    // Explicit backcast: sigma2[0] = omega + (alpha + beta) * backcast
    model.backcast = 14.0 / 3.0;
    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_OK, "filter with backcast OK");
    s2_0 = 0.2 + 0.8 * (14.0 / 3.0);
    s2_1 = 0.2 + 0.3 * 1.0 + 0.5 * s2_0;
    ASSERT_NEAR(sigma[0], sqrt(s2_0), 1e-12, "filter step 0 (explicit backcast)");
    ASSERT_NEAR(sigma[1], sqrt(s2_1), 1e-12, "filter step 1 (explicit backcast)");
    PASS("garch filter known answer");
}

static int test_garch_filter_invalid_inputs(void) {
    mlr_garch model = {.omega = 0.2, .alpha = 0.3, .beta = 0.5, .converged = 1, .backcast = 1.0};
    double returns[] = {1.0, -2.0, 3.0};
    double sigma[3];

    ASSERT(mlr_garch_filter(NULL, returns, 3, sigma) == MLR_EINVAL, "NULL model -> EINVAL");
    ASSERT(mlr_garch_filter(&model, NULL, 3, sigma) == MLR_EINVAL, "NULL returns -> EINVAL");
    ASSERT(mlr_garch_filter(&model, returns, 3, NULL) == MLR_EINVAL, "NULL sigma_out -> EINVAL");
    ASSERT(mlr_garch_filter(&model, returns, 0, sigma) == MLR_EINVAL, "n=0 -> EINVAL");

    model.backcast = -1.0;
    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_EINVAL, "negative backcast -> EINVAL");
    model.backcast = MLR_NAN;
    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_EINVAL, "NAN backcast -> EINVAL");
    model.backcast = 1.0;

    model.omega = INFINITY;
    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_EINVAL, "omega=Inf -> EINVAL");
    ASSERT(mlr_garch_forecast(&model, 3, sigma) == MLR_EINVAL, "forecast omega=Inf -> EINVAL");
    model.omega = 0.0;
    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_EINVAL, "omega=0 -> EINVAL");
    model.omega = 0.2;
    model.alpha = 0.5;
    ASSERT(mlr_garch_filter(&model, returns, 3, sigma) == MLR_EINVAL, "alpha+beta>=1 -> EINVAL");

    // Valid but extreme parameters that overflow the recursion fail loudly
    mlr_garch extreme = {.omega = 1e308, .alpha = 0.3, .beta = 0.5, .converged = 1, .backcast = 1e308};
    ASSERT(mlr_garch_filter(&extreme, returns, 3, sigma) == MLR_EDOMAIN, "overflowing recursion -> EDOMAIN");
    extreme.sigma2_next = 1.0;   // uncond = 1e308 / 0.2 overflows
    ASSERT(mlr_garch_forecast(&extreme, 3, sigma) == MLR_EDOMAIN, "overflowing forecast -> EDOMAIN");
    PASS("garch filter invalid inputs");
}

static int test_garch_filter_missing_data(void) {
    // sigma_out[t] was determined before returns[t] is seen, so a missing
    // return does not change it; the recursion then steps its own forecast
    mlr_garch model = {.omega = 0.2, .alpha = 0.3, .beta = 0.5, .converged = 1, .backcast = 2.0};
    double returns[] = {1.0, MLR_NAN, -1.0, 0.5};
    double sigma[4];

    ASSERT(mlr_garch_filter(&model, returns, 4, sigma) == MLR_OK, "filter with NAN return OK");
    double s2_0 = 0.2 + 0.8 * 2.0;
    double s2_1 = 0.2 + 0.3 * 1.0 + 0.5 * s2_0;
    double s2_2 = 0.2 + 0.8 * s2_1;
    double s2_3 = 0.2 + 0.3 * 1.0 + 0.5 * s2_2;
    ASSERT_NEAR(sigma[0], sqrt(s2_0), 1e-12, "missing-data step 0");
    ASSERT_NEAR(sigma[1], sqrt(s2_1), 1e-12, "forecast at the missing bar is still emitted");
    ASSERT_NEAR(sigma[2], sqrt(s2_2), 1e-12, "recursion steps its forecast over the gap");
    ASSERT_NEAR(sigma[3], sqrt(s2_3), 1e-12, "missing-data step 3");

    // A finite return whose square overflows is treated the same way
    double huge[] = {1.0, 1e200, -1.0, 0.5};
    double sigma_huge[4];
    ASSERT(mlr_garch_filter(&model, huge, 4, sigma_huge) == MLR_OK, "filter with overflowing return OK");
    for (size_t t = 0; t < 4; t++) {
        ASSERT(sigma_huge[t] == sigma[t], "overflowing return behaves as missing");
    }
    PASS("garch filter missing data");
}

static int test_garch_forecast(void) {
    mlr_garch model = {.omega = 0.2, .alpha = 0.3, .beta = 0.5, .sigma2_next = 2.5, .converged = 1};
    double sigma[1000];

    ASSERT(mlr_garch_forecast(&model, 1000, sigma) == MLR_OK, "garch_forecast returns MLR_OK");
    ASSERT_NEAR(sigma[0], sqrt(2.5), 1e-12, "h=1 equals sqrt(sigma2_next)");
    // uncond = omega/(1-alpha-beta) = 1; h=2: 1 + 0.8*(2.5-1)
    ASSERT_NEAR(sigma[1], sqrt(1.0 + 0.8 * 1.5), 1e-12, "h=2 decays toward unconditional");
    ASSERT_NEAR(sigma[999], 1.0, 1e-6, "long horizon converges to unconditional vol");

    ASSERT(mlr_garch_forecast(&model, 0, sigma) == MLR_EINVAL, "horizon=0 -> EINVAL");
    ASSERT(mlr_garch_forecast(NULL, 1, sigma) == MLR_EINVAL, "NULL model -> EINVAL");
    ASSERT(mlr_garch_forecast(&model, 1, NULL) == MLR_EINVAL, "NULL sigma_out -> EINVAL");
    model.sigma2_next = MLR_NAN;
    ASSERT(mlr_garch_forecast(&model, 1, sigma) == MLR_EINVAL, "NAN sigma2_next -> EINVAL");
    model.sigma2_next = -1.0;
    ASSERT(mlr_garch_forecast(&model, 1, sigma) == MLR_EINVAL, "negative sigma2_next -> EINVAL");
    model.sigma2_next = 0.0;
    ASSERT(mlr_garch_forecast(&model, 1, sigma) == MLR_EINVAL, "zero sigma2_next (unset hand-built model) -> EINVAL");
    model.sigma2_next = 2.5;
    model.alpha = -0.1;
    ASSERT(mlr_garch_forecast(&model, 1, sigma) == MLR_EINVAL, "bad params -> EINVAL");
    PASS("garch forecast");
}

// Same NLL the fitter minimizes (constants dropped), replicated independently.
// Seed: sigma2[0] = omega + (alpha + beta) * mean(r^2)
static double test_nll(const double *r, size_t n, double omega, double alpha, double beta) {
    double var0 = 0.0;
    for (size_t t = 0; t < n; t++) var0 += r[t] * r[t];
    var0 /= (double)n;

    double s2 = omega + (alpha + beta) * var0;
    double nll = 0.0;
    for (size_t t = 0; t < n; t++) {
        nll += log(s2) + (r[t] * r[t]) / s2;
        s2 = omega + alpha * r[t] * r[t] + beta * s2;
    }
    return 0.5 * nll;
}

// GARCH(1,1) path started at its unconditional variance.
// Mirrored exactly in tests/reference/garch_arch_reference.py.
static void simulate_garch(unsigned long long seed, double omega, double alpha, double beta,
                           size_t n, double *returns) {
    unsigned long long state = seed;
    double s2 = omega / (1.0 - alpha - beta);
    for (size_t t = 0; t < n; t++) {
        returns[t] = sqrt(s2) * test_lcg_gauss(&state);
        s2 = omega + alpha * returns[t] * returns[t] + beta * s2;
    }
}

static int test_garch_fit_recovers_simulation(void) {
    enum { N = 2000 };
    static double returns[N];
    const double true_omega = 2e-6, true_alpha = 0.10, true_beta = 0.85;

    simulate_garch(12345, true_omega, true_alpha, true_beta, N, returns);

    mlr_garch model;
    ASSERT(mlr_garch_fit(returns, N, &model) == MLR_OK, "garch_fit returns MLR_OK");
    ASSERT(model.converged, "garch_fit converges");
    ASSERT(model.omega > 0.0 && model.alpha >= 0.0 && model.beta >= 0.0, "constraints hold");
    ASSERT(model.alpha + model.beta < 0.9999, "persistence below the bound");

    // The fit is at least as good as the truth under the same objective
    double nll_fit = test_nll(returns, N, model.omega, model.alpha, model.beta);
    double nll_true = test_nll(returns, N, true_omega, true_alpha, true_beta);
    ASSERT(nll_fit <= nll_true + 1e-9, "fitted NLL does not exceed true-parameter NLL");
    ASSERT_NEAR(model.loglik, -nll_fit, 1e-6 * fabs(nll_fit), "loglik field matches the objective");

    double var0 = 0.0;
    for (size_t t = 0; t < N; t++) var0 += returns[t] * returns[t];
    var0 /= (double)N;
    ASSERT_NEAR(model.backcast, var0, 1e-15 * var0, "fit stores mean(r^2) as backcast");
    PASS("garch fit recovers simulated parameters");
}

// Reference values from the Python `arch` package (7.2.0) fitting the identical
// simulated sample with the identical backcast, so both sides maximize the same
// function. Regenerate with tests/reference/garch_arch_reference.py.
typedef struct {
    unsigned long long seed;
    double omega, alpha, beta;
    size_t n;
    double ref_omega, ref_alpha, ref_beta, ref_loglik;
} arch_case;

static int test_garch_fit_matches_arch(void) {
    static const arch_case cases[] = {
        {12345, 2e-6, 0.10, 0.85, 2000,
         1.355607201170e-06, 0.081035203406, 0.883501535165, 9259.949082},
        {777, 5e-6, 0.05, 0.93, 1500,
         5.176457051708e-06, 0.050852929935, 0.925219134086, 5606.487198},
    };
    enum { MAX_N = 2000 };
    static double returns[MAX_N];

    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const arch_case *k = &cases[c];
        ASSERT(k->n <= MAX_N, "reference sample fits the buffer");
        simulate_garch(k->seed, k->omega, k->alpha, k->beta, k->n, returns);

        mlr_garch model;
        ASSERT(mlr_garch_fit(returns, k->n, &model) == MLR_OK, "arch-parity fit OK");
        ASSERT(model.converged, "arch-parity fit converged");
        ASSERT_NEAR(model.alpha, k->ref_alpha, 1e-5, "alpha matches arch");
        ASSERT_NEAR(model.beta, k->ref_beta, 1e-5, "beta matches arch");
        ASSERT_NEAR(model.omega / k->ref_omega, 1.0, 1e-4, "omega matches arch");
        ASSERT_NEAR(model.loglik, k->ref_loglik, 1e-8 * fabs(k->ref_loglik), "loglik matches arch (same objective)");
    }
    PASS("garch fit matches arch reference");
}

static int test_garch_fit_scale_invariance(void) {
    // alpha and beta are dimensionless; omega scales with the variance
    enum { N = 1000 };
    static double returns[N], scaled[N];
    simulate_garch(2024, 2e-6, 0.10, 0.85, N, returns);

    mlr_garch base;
    ASSERT(mlr_garch_fit(returns, N, &base) == MLR_OK, "base fit OK");
    ASSERT(base.converged, "base fit converged");

    double scales[] = {1e-8, 1e-3, 1e3, 1e6};
    for (size_t s = 0; s < sizeof scales / sizeof scales[0]; s++) {
        for (size_t t = 0; t < N; t++) scaled[t] = returns[t] * scales[s];
        mlr_garch m;
        ASSERT(mlr_garch_fit(scaled, N, &m) == MLR_OK, "scaled fit OK");
        ASSERT(m.converged, "scaled fit converged");
        // 1e-6 is the floating-point floor: the objective sums a thousand
        // log terms whose magnitude depends on the scale
        ASSERT_NEAR(m.alpha, base.alpha, 1e-6, "alpha is scale invariant");
        ASSERT_NEAR(m.beta, base.beta, 1e-6, "beta is scale invariant");
        ASSERT_NEAR(m.omega / (base.omega * scales[s] * scales[s]), 1.0, 1e-5, "omega scales with variance");
    }
    PASS("garch fit scale invariance");
}

static int test_garch_fit_no_arch_effect(void) {
    // iid returns: the maximum is on the boundary alpha = 0. The optimizer
    // must converge there, not run to its iteration cap.
    enum { N = 2000 };
    static double returns[N];
    unsigned long long state = 5;
    for (size_t t = 0; t < N; t++) returns[t] = 0.01 * test_lcg_gauss(&state);

    mlr_garch model;
    ASSERT(mlr_garch_fit(returns, N, &model) == MLR_OK, "fit on iid returns OK");
    ASSERT(model.converged, "fit on iid returns converges");
    ASSERT(model.alpha < 0.05, "alpha near the boundary on iid returns (sampling noise allows a few 1e-3)");
    ASSERT(mlr_isfinite(model.loglik), "loglik finite");
    PASS("garch fit with no ARCH effect");
}

static int test_garch_fit_invalid_inputs(void) {
    double zeros[100] = {0.0};
    double small[50] = {0.01, -0.01};
    mlr_garch model;

    ASSERT(mlr_garch_fit(zeros, 100, &model) == MLR_EDOMAIN, "zero-variance returns -> EDOMAIN");
    ASSERT(mlr_garch_fit(small, 50, &model) == MLR_EINVAL, "n < 100 -> EINVAL");
    ASSERT(mlr_garch_fit(NULL, 100, &model) == MLR_EINVAL, "NULL returns -> EINVAL");
    ASSERT(mlr_garch_fit(zeros, 100, NULL) == MLR_EINVAL, "NULL model -> EINVAL");

    // Denormal variance is refused; a variance just above DBL_MIN still fits
    // to the same alpha and beta as the unscaled series
    static double tiny[2000], base[2000];
    simulate_garch(12345, 2e-6, 0.10, 0.85, 2000, base);
    mlr_garch ref;
    ASSERT(mlr_garch_fit(base, 2000, &ref) == MLR_OK, "reference fit OK");
    for (size_t t = 0; t < 2000; t++) tiny[t] = base[t] * 1e-158;
    ASSERT(mlr_garch_fit(tiny, 2000, &model) == MLR_EDOMAIN, "denormal variance -> EDOMAIN");
    for (size_t t = 0; t < 2000; t++) tiny[t] = base[t] * 1e-150;
    ASSERT(mlr_garch_fit(tiny, 2000, &model) == MLR_OK, "variance above DBL_MIN fits");
    ASSERT_NEAR(model.alpha, ref.alpha, 1e-6, "alpha unchanged at scale 1e-150");
    ASSERT_NEAR(model.beta, ref.beta, 1e-6, "beta unchanged at scale 1e-150");

    zeros[5] = MLR_NAN;
    ASSERT(mlr_garch_fit(zeros, 100, &model) == MLR_EINVAL, "non-finite return -> EINVAL");
    zeros[5] = 1e200;
    ASSERT(mlr_garch_fit(zeros, 100, &model) == MLR_EDOMAIN, "overflowing variance -> EDOMAIN");
    PASS("garch fit invalid inputs");
}

static int test_garch_filter_no_lookahead(void) {
    enum { N = 600, HALF = 300 };
    static double returns[N], full[N], prefix[HALF];

    simulate_garch(99, 3e-6, 0.08, 0.90, N, returns);

    mlr_garch model;
    ASSERT(mlr_garch_fit(returns, HALF, &model) == MLR_OK, "fit on first half OK");

    // sigma[t] depends only on returns before t: filtering the whole series
    // and filtering a prefix agree exactly on the prefix
    ASSERT(mlr_garch_filter(&model, returns, N, full) == MLR_OK, "filter full OK");
    ASSERT(mlr_garch_filter(&model, returns, HALF, prefix) == MLR_OK, "filter prefix OK");
    for (size_t t = 0; t < HALF; t++) {
        ASSERT(full[t] == prefix[t], "filter output is prefix-stable");
    }

    // Filtering the fit sample ends exactly where the fit says it does
    double last = returns[HALF - 1];
    double s2_next = model.omega + model.alpha * last * last + model.beta * prefix[HALF - 1] * prefix[HALF - 1];
    ASSERT_NEAR(s2_next, model.sigma2_next, 1e-12 * model.sigma2_next,
                "filter over the fit sample reproduces sigma2_next");

    // Continuing from sigma2_next onto the second half equals the tail of the
    // full filter, bit for bit; filtering the second half alone does not
    static double cont[HALF], alone[HALF];
    ASSERT(mlr_garch_filter_from(&model, model.sigma2_next, returns + HALF, HALF, cont) == MLR_OK, "filter_from OK");
    for (size_t t = 0; t < HALF; t++) {
        ASSERT(cont[t] == full[HALF + t], "filter_from(sigma2_next) continues the full filter exactly");
    }
    ASSERT(mlr_garch_filter(&model, returns + HALF, HALF, alone) == MLR_OK, "filter of the tail alone OK");
    ASSERT(alone[0] != full[HALF], "filtering the tail alone restarts from the backcast");

    // sigma2_next and the filter share one recursion step, so the tail is
    // bit-identical for every fit sample, checked across seeds
    for (unsigned long long seed = 1; seed <= 60; seed++) {
        simulate_garch(seed, 3e-6, 0.08, 0.90, N, returns);
        mlr_garch m;
        ASSERT(mlr_garch_fit(returns, HALF, &m) == MLR_OK, "seeded fit OK");
        ASSERT(mlr_garch_filter(&m, returns, N, full) == MLR_OK, "seeded full filter OK");
        ASSERT(mlr_garch_filter_from(&m, m.sigma2_next, returns + HALF, HALF, cont) == MLR_OK, "seeded continuation OK");
        for (size_t t = 0; t < HALF; t++) {
            ASSERT(cont[t] == full[HALF + t], "continuation is bit-identical for every seed");
        }
    }

    ASSERT(mlr_garch_filter_from(&model, 0.0, returns, HALF, cont) == MLR_EINVAL, "sigma2_first=0 -> EINVAL");
    ASSERT(mlr_garch_filter_from(&model, MLR_NAN, returns, HALF, cont) == MLR_EINVAL, "NAN sigma2_first -> EINVAL");
    ASSERT(mlr_garch_filter_from(&model, 1.0, returns, 0, cont) == MLR_EINVAL, "n=0 -> EINVAL");
    PASS("garch filter no lookahead");
}

static int test_volatility_timing_alignment(void) {
    // Every volatility forecast in the library responds to a shock the bar
    // AFTER it happens, never on the same bar, so sigma[t] can size a
    // position held over bar t
    enum { N = 30, K = 15 };
    double calm[N], shocked[N], ewma_calm[N], ewma_shock[N], garch_calm[N], garch_shock[N];
    for (size_t t = 0; t < N; t++) calm[t] = shocked[t] = 0.005;
    shocked[K] = 0.05;

    ASSERT(mlr_ewma_vol(calm, N, 0.94, ewma_calm) == MLR_OK, "ewma OK");
    ASSERT(mlr_ewma_vol(shocked, N, 0.94, ewma_shock) == MLR_OK, "ewma OK");
    mlr_garch model = {.omega = 1e-6, .alpha = 0.1, .beta = 0.85, .converged = 1, .backcast = 0.0};
    ASSERT(mlr_garch_filter(&model, calm, N, garch_calm) == MLR_OK, "garch filter OK");
    ASSERT(mlr_garch_filter(&model, shocked, N, garch_shock) == MLR_OK, "garch filter OK");

    for (size_t t = 0; t <= K; t++) {
        ASSERT(ewma_shock[t] == ewma_calm[t] || (mlr_isnan(ewma_shock[t]) && mlr_isnan(ewma_calm[t])),
               "ewma output through the shock bar is unaffected by the shock");
        ASSERT(garch_shock[t] == garch_calm[t], "garch output through the shock bar is unaffected by the shock");
    }
    ASSERT(ewma_shock[K + 1] > 2.0 * ewma_calm[K + 1], "ewma responds the bar after the shock");
    ASSERT(garch_shock[K + 1] > 2.0 * garch_calm[K + 1], "garch responds the bar after the shock");
    PASS("volatility timing alignment");
}

int test_vol(void) {
    int failures = 0;
    failures += test_parkinson_known_answer();
    failures += test_garman_klass_known_answer();
    failures += test_garch_filter_known_answer();
    failures += test_garch_filter_invalid_inputs();
    failures += test_garch_filter_missing_data();
    failures += test_garch_forecast();
    failures += test_garch_fit_recovers_simulation();
    failures += test_garch_fit_matches_arch();
    failures += test_garch_fit_scale_invariance();
    failures += test_garch_fit_no_arch_effect();
    failures += test_garch_fit_invalid_inputs();
    failures += test_garch_filter_no_lookahead();
    failures += test_volatility_timing_alignment();
    return failures;
}
