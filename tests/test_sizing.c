#include "mlrisk/sizing.h"
#include "test_util.h"

#define TOL 1e-9

static int test_vol_target_position_basic(void) {
    double sigma[] = {0.01, 0.02, 0.015};
    double price[] = {100.0, 100.0, 100.0};
    double out[3];

    ASSERT(mlr_vol_target_position(sigma, 0.01, 10000.0, price, 2.0, 3, out) == MLR_OK,
           "vol_target_position returns MLR_OK");
    ASSERT_NEAR(out[0], 100.0, TOL, "(0.01/0.01)*(10000/100)");
    ASSERT_NEAR(out[1], 50.0, TOL, "(0.01/0.02)*(10000/100)");
    ASSERT_NEAR(out[2], 10000.0 / 150.0, TOL, "(0.01/0.015)*(10000/100)");
    PASS("vol_target_position basic");
}

static int test_vol_target_position_bad_sigma_or_price(void) {
    double sigma[] = {0.01, MLR_NAN, -0.01, INFINITY, 0.01, 0.01, 0.01};
    double price[] = {100.0, 100.0, 100.0, 100.0, MLR_NAN, INFINITY, 0.0};
    double out[7];

    ASSERT(mlr_vol_target_position(sigma, 0.01, 10000.0, price, 2.0, 7, out) == MLR_OK,
           "bad elements do not fail the call");
    ASSERT_NEAR(out[0], 100.0, TOL, "good element sized");
    for (size_t i = 1; i < 7; i++) {
        ASSERT(out[i] == 0.0, "NAN/negative/Inf sigma or NAN/Inf/zero price gives 0");
    }
    PASS("vol_target_position bad sigma or price");
}

static int test_vol_target_position_leverage_cap(void) {
    double sigma[] = {0.001};
    double price[] = {100.0};
    double out[1];

    // Uncapped: (0.01/0.001)*(10000/100) = 1000; cap: 2*10000/100 = 200
    ASSERT(mlr_vol_target_position(sigma, 0.01, 10000.0, price, 2.0, 1, out) == MLR_OK, "OK");
    ASSERT_NEAR(out[0], 200.0, TOL, "position capped at max_leverage * equity / price");
    PASS("vol_target_position leverage cap");
}

static int test_vol_target_position_as_risk_cap(void) {
    // Risk capping is the same formula with target_vol read as the risk cap
    double sigma[] = {0.01, 0.02};
    double price[] = {100.0, 100.0};
    double out[2];

    ASSERT(mlr_vol_target_position(sigma, 0.02, 10000.0, price, 2.0, 2, out) == MLR_OK, "OK");
    ASSERT_NEAR(out[0], 200.0, TOL, "(0.02/0.01)*(10000/100)");
    ASSERT_NEAR(out[1], 100.0, TOL, "(0.02/0.02)*(10000/100)");
    PASS("vol_target_position as risk cap");
}

static int test_vol_target_position_invalid_inputs(void) {
    double sigma[] = {0.01};
    double price[] = {100.0};
    double out[1];

    ASSERT(mlr_vol_target_position(NULL, 0.01, 1e4, price, 2.0, 1, out) == MLR_EINVAL, "NULL sigma");
    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e4, NULL, 2.0, 1, out) == MLR_EINVAL, "NULL price");
    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e4, price, 2.0, 1, NULL) == MLR_EINVAL, "NULL out");
    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e4, price, 2.0, 0, out) == MLR_EINVAL, "n=0");

    ASSERT(mlr_vol_target_position(sigma, -0.01, 1e4, price, 2.0, 1, out) == MLR_EINVAL, "negative target_vol");
    ASSERT(mlr_vol_target_position(sigma, 0.0, 1e4, price, 2.0, 1, out) == MLR_EINVAL, "zero target_vol");
    ASSERT(mlr_vol_target_position(sigma, MLR_NAN, 1e4, price, 2.0, 1, out) == MLR_EINVAL, "NAN target_vol");

    ASSERT(mlr_vol_target_position(sigma, 0.01, -1e3, price, 2.0, 1, out) == MLR_EINVAL, "negative equity");
    ASSERT(mlr_vol_target_position(sigma, 0.01, MLR_NAN, price, 2.0, 1, out) == MLR_EINVAL, "NAN equity");
    ASSERT(mlr_vol_target_position(sigma, 0.01, INFINITY, price, 2.0, 1, out) == MLR_EINVAL, "Inf equity");

    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e4, price, -1.0, 1, out) == MLR_EINVAL, "negative leverage");
    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e4, price, INFINITY, 1, out) == MLR_EINVAL, "Inf leverage");
    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e4, price, MLR_NAN, 1, out) == MLR_EINVAL, "NAN leverage");
    // Finite equity and leverage whose product overflows would disable the cap
    ASSERT(mlr_vol_target_position(sigma, 0.01, 1e200, price, 1e200, 1, out) == MLR_EINVAL, "overflowing cap -> EINVAL");
    PASS("vol_target_position invalid inputs");
}

static int test_kelly_fraction(void) {
    // mean = 0.025, sample var = 0.0075 -> full Kelly 10/3, half Kelly 5/3
    double returns[] = {0.1, -0.05, 0.1, -0.05};
    double f = 0.0;

    ASSERT(mlr_kelly_fraction(returns, 4, 1.0, &f) == MLR_OK, "kelly OK");
    ASSERT_NEAR(f, 10.0 / 3.0, TOL, "full Kelly");
    ASSERT(mlr_kelly_fraction(returns, 4, 0.5, &f) == MLR_OK, "half kelly OK");
    ASSERT_NEAR(f, 5.0 / 3.0, TOL, "half Kelly");

    double losing[] = {-0.1, 0.05, -0.1, 0.05};
    ASSERT(mlr_kelly_fraction(losing, 4, 1.0, &f) == MLR_OK, "negative-edge kelly OK");
    ASSERT_NEAR(f, -10.0 / 3.0, TOL, "negative edge gives negative fraction");

    double constant[] = {0.01, 0.01, 0.01};
    ASSERT(mlr_kelly_fraction(constant, 3, 1.0, &f) == MLR_EDOMAIN, "zero variance -> EDOMAIN");
    ASSERT(mlr_kelly_fraction(returns, 1, 1.0, &f) == MLR_EINVAL, "n<2 -> EINVAL");
    ASSERT(mlr_kelly_fraction(returns, 4, 0.0, &f) == MLR_EINVAL, "fraction=0 -> EINVAL");
    ASSERT(mlr_kelly_fraction(returns, 4, MLR_NAN, &f) == MLR_EINVAL, "NAN fraction -> EINVAL");
    ASSERT(mlr_kelly_fraction(NULL, 4, 1.0, &f) == MLR_EINVAL, "NULL returns -> EINVAL");
    ASSERT(mlr_kelly_fraction(returns, 4, 1.0, NULL) == MLR_EINVAL, "NULL f_out -> EINVAL");

    double with_nan[] = {0.1, MLR_NAN, 0.1};
    ASSERT(mlr_kelly_fraction(with_nan, 3, 1.0, &f) == MLR_EINVAL, "non-finite return -> EINVAL");

    // Finite inputs whose sums overflow must not produce a NAN estimate
    double huge[] = {1e308, 1e308, -1e308};
    ASSERT(mlr_kelly_fraction(huge, 3, 1.0, &f) == MLR_EDOMAIN, "overflowing sums -> EDOMAIN");
    // Finite mean, overflowing variance: refused rather than rounded to 0
    double wide[] = {3e154, -1e154, 3e154, -1e154};
    ASSERT(mlr_kelly_fraction(wide, 4, 1.0, &f) == MLR_EDOMAIN, "overflowing variance -> EDOMAIN");
    PASS("kelly fraction");
}

static int test_drawdown_scale(void) {
    // Peaks 100,110,110,110,120; drawdowns 0,0,0.1,0.05,0
    double equity[] = {100.0, 110.0, 99.0, 104.5, 120.0};
    double scale[5];

    ASSERT(mlr_drawdown_scale(equity, 5, 0.2, scale) == MLR_OK, "drawdown_scale OK");
    ASSERT_NEAR(scale[0], 1.0, TOL, "no drawdown at start");
    ASSERT_NEAR(scale[1], 1.0, TOL, "no drawdown at new peak");
    ASSERT_NEAR(scale[2], 0.5, TOL, "10% drawdown of a 20% cap");
    ASSERT_NEAR(scale[3], 0.75, TOL, "5% drawdown of a 20% cap");
    ASSERT_NEAR(scale[4], 1.0, TOL, "new peak");

    double crash[] = {100.0, 50.0};
    ASSERT(mlr_drawdown_scale(crash, 2, 0.2, scale) == MLR_OK, "crash path OK");
    ASSERT(scale[1] == 0.0, "drawdown past max_dd scales to 0");

    // Alignment: scale[t] reflects equity[t], the close at the end of
    // period t. The position held over period t therefore takes scale[t-1];
    // on the bar of the loss that is still 1, and only the next bar is cut.
    double path[] = {100.0, 100.0, 50.0, 55.0};
    ASSERT(mlr_drawdown_scale(path, 4, 0.2, scale) == MLR_OK, "path OK");
    ASSERT(scale[1] == 1.0 && scale[2] == 0.0, "scale[t] is computed from equity[t] (contemporaneous)");
    double applied[4] = {1.0, scale[0], scale[1], scale[2]};   // scale for position[t] is scale[t-1]
    ASSERT(applied[2] == 1.0 && applied[3] == 0.0, "lagged application cuts exposure the bar after the loss, not on it");

    ASSERT(mlr_drawdown_scale(equity, 5, 0.0, scale) == MLR_EINVAL, "max_dd=0 -> EINVAL");
    ASSERT(mlr_drawdown_scale(equity, 5, 1.5, scale) == MLR_EINVAL, "max_dd>1 -> EINVAL");
    ASSERT(mlr_drawdown_scale(equity, 5, MLR_NAN, scale) == MLR_EINVAL, "NAN max_dd -> EINVAL");
    ASSERT(mlr_drawdown_scale(equity, 0, 0.2, scale) == MLR_EINVAL, "n=0 -> EINVAL");
    ASSERT(mlr_drawdown_scale(NULL, 5, 0.2, scale) == MLR_EINVAL, "NULL equity -> EINVAL");
    ASSERT(mlr_drawdown_scale(equity, 5, 0.2, NULL) == MLR_EINVAL, "NULL scale_out -> EINVAL");

    double bad[] = {100.0, 0.0};
    ASSERT(mlr_drawdown_scale(bad, 2, 0.2, scale) == MLR_EDOMAIN, "non-positive equity -> EDOMAIN");
    bad[1] = MLR_NAN;
    ASSERT(mlr_drawdown_scale(bad, 2, 0.2, scale) == MLR_EINVAL, "NAN equity -> EINVAL");
    PASS("drawdown scale");
}

int test_sizing(void) {
    int failures = 0;
    failures += test_vol_target_position_basic();
    failures += test_vol_target_position_bad_sigma_or_price();
    failures += test_vol_target_position_leverage_cap();
    failures += test_vol_target_position_as_risk_cap();
    failures += test_vol_target_position_invalid_inputs();
    failures += test_kelly_fraction();
    failures += test_drawdown_scale();
    return failures;
}
