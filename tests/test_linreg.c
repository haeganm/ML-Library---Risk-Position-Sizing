#include "mlrisk/linreg.h"
#include "test_util.h"
#include <stdint.h>

#define TOL 1e-9

static int test_linreg_exact_1d(void) {
    // y = 2x + 1 exactly
    double X[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 1) == MLR_OK, "init OK");

    ASSERT(mlr_linreg_fit(X, y, 5, 1, 0.0, &model) == MLR_OK, "fit OK");
    ASSERT_NEAR(model.w[0], 2.0, TOL, "slope");
    ASSERT_NEAR(model.b, 1.0, TOL, "intercept");
    ASSERT(model.ridge == 0.0, "model records the ridge used");

    double X_test[] = {5.0, 6.0};
    double pred[2];
    ASSERT(mlr_linreg_predict(X_test, 2, 1, &model, pred) == MLR_OK, "predict OK");
    ASSERT_NEAR(pred[0], 11.0, TOL, "prediction at 5");
    ASSERT_NEAR(pred[1], 13.0, TOL, "prediction at 6");

    // Ridge shrinks the slope toward zero: w = Sxy / (Sxx + ridge), b = mean(y) - mean(x) w.
    // Compared against a double variable, not the literal: under x87
    // excess precision (32-bit x86) a literal is evaluated in long double.
    const double ridge = 0.01;
    ASSERT(mlr_linreg_fit(X, y, 5, 1, ridge, &model) == MLR_OK, "ridge fit OK");
    double w = 20.0 / (10.0 + ridge);
    ASSERT_NEAR(model.w[0], w, TOL, "ridge slope");
    ASSERT_NEAR(model.b, 5.0 - 2.0 * w, TOL, "ridge intercept");
    ASSERT(model.ridge == ridge, "model records the ridge used");

    mlr_lin_model_free(&model);
    PASS("linreg exact 1D");
}

static int test_linreg_exact_2d(void) {
    // y = x1 + 2 x2 + 3 exactly
    double X[] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 2.0
    };
    double y[] = {3.0, 4.0, 5.0, 6.0, 9.0};
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 2) == MLR_OK, "init OK");

    ASSERT(mlr_linreg_fit(X, y, 5, 2, 0.0, &model) == MLR_OK, "fit OK");
    ASSERT_NEAR(model.w[0], 1.0, TOL, "w1");
    ASSERT_NEAR(model.w[1], 2.0, TOL, "w2");
    ASSERT_NEAR(model.b, 3.0, TOL, "intercept");

    double X_test[] = {3.0, -1.0};
    double pred[1];
    ASSERT(mlr_linreg_predict(X_test, 1, 2, &model, pred) == MLR_OK, "predict OK");
    ASSERT_NEAR(pred[0], 4.0, TOL, "prediction");

    mlr_lin_model_free(&model);
    PASS("linreg exact 2D");
}

static int test_linreg_ridge_on_rank_deficient(void) {
    // Duplicate columns: singular without ridge. With ridge, symmetry gives
    // w1 = w2 = Sxy / (2 Sxx + ridge) and b = mean(y) - 2 mean(x) w.
    // Column {1,2,3,4}: Sxx = 5, Sxy = 5 against y = {1,2,3,4}.
    double X[] = {
        1.0, 1.0,
        2.0, 2.0,
        3.0, 3.0,
        4.0, 4.0
    };
    double y[] = {1.0, 2.0, 3.0, 4.0};
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 2) == MLR_OK, "init OK");

    ASSERT(mlr_linreg_fit(X, y, 4, 2, 0.0, &model) == MLR_EDOMAIN, "singular without ridge -> EDOMAIN");

    ASSERT(mlr_linreg_fit(X, y, 4, 2, 1.0, &model) == MLR_OK, "ridge makes it solvable");
    double w = 5.0 / 11.0;
    ASSERT_NEAR(model.w[0], w, TOL, "w1 = Sxy/(2Sxx+ridge)");
    ASSERT_NEAR(model.w[1], w, TOL, "w2 = Sxy/(2Sxx+ridge)");
    ASSERT_NEAR(model.b, 2.5 - 5.0 * w, TOL, "intercept");

    mlr_lin_model_free(&model);
    PASS("ridge on rank-deficient features");
}

static int test_linreg_scale_invariance(void) {
    // y = 2e12 x + 1 with features at 1e-12: the singularity threshold is
    // relative to the matrix, so a perfectly conditioned tiny system passes
    double X[] = {0.0, 1e-12, 2e-12, 3e-12, 4e-12};
    double y[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 1) == MLR_OK, "init OK");

    ASSERT(mlr_linreg_fit(X, y, 5, 1, 0.0, &model) == MLR_OK, "tiny-scale fit OK");
    ASSERT_NEAR(model.w[0] / 2e12, 1.0, 1e-6, "slope at tiny scale");

    double Xb[] = {0.0, 1e12, 2e12, 3e12, 4e12};
    ASSERT(mlr_linreg_fit(Xb, y, 5, 1, 0.0, &model) == MLR_OK, "huge-scale fit OK");
    ASSERT_NEAR(model.w[0] * 1e12, 2.0, 1e-6, "slope at huge scale");

    // Finite features whose squares overflow cannot be fit
    double Xo[] = {0.0, 1e200, 2e200, 3e200, 4e200};
    ASSERT(mlr_linreg_fit(Xo, y, 5, 1, 0.0, &model) == MLR_EDOMAIN, "overflowing normal matrix -> EDOMAIN");

    mlr_lin_model_free(&model);
    PASS("linreg scale invariance");
}

static int test_linreg_invalid_inputs(void) {
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 2.0};
    double out[2];
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 2) == MLR_OK, "init OK");

    ASSERT(mlr_linreg_fit(NULL, y, 2, 2, 0.0, &model) == MLR_EINVAL, "NULL X");
    ASSERT(mlr_linreg_fit(X, NULL, 2, 2, 0.0, &model) == MLR_EINVAL, "NULL y");
    ASSERT(mlr_linreg_fit(X, y, 0, 2, 0.0, &model) == MLR_EINVAL, "n=0");
    ASSERT(mlr_linreg_fit(X, y, 2, 3, 0.0, &model) == MLR_EINVAL, "d != model->d");
    ASSERT(mlr_linreg_fit(X, y, 2, 2, -1.0, &model) == MLR_EINVAL, "negative ridge");
    ASSERT(mlr_linreg_fit(X, y, 2, 2, MLR_NAN, &model) == MLR_EINVAL, "NAN ridge");

    double X_nan[] = {1.0, MLR_NAN, 3.0, 4.0};
    ASSERT(mlr_linreg_fit(X_nan, y, 2, 2, 1.0, &model) == MLR_EINVAL, "NAN in X");
    double y_inf[] = {1.0, INFINITY};
    ASSERT(mlr_linreg_fit(X, y_inf, 2, 2, 1.0, &model) == MLR_EINVAL, "Inf in y");

    ASSERT(mlr_linreg_predict(NULL, 1, 2, &model, out) == MLR_EINVAL, "predict NULL X");
    ASSERT(mlr_linreg_predict(X, 1, 3, &model, out) == MLR_EINVAL, "predict d mismatch");
    ASSERT(mlr_linreg_predict(X, 0, 2, &model, out) == MLR_EINVAL, "predict n=0");

    mlr_lin_model_free(&model);
    ASSERT(mlr_linreg_fit(X, y, 2, 2, 0.0, &model) == MLR_EINVAL, "fit on a freed model");
    ASSERT(mlr_linreg_predict(X, 1, 2, &model, out) == MLR_EINVAL, "predict on a freed model");
    PASS("linreg invalid inputs");
}

static int test_linreg_ill_conditioned(void) {
    // Two nearly collinear columns: x2 = x1 + 1e-7 * z, condition number
    // around 1e8. The normal equations lose this problem (cond^2 * eps > 1);
    // QR on the design keeps about 8 digits. y = 3 x1 - 2 x2 + 1.
    enum { N = 40 };
    static double X[N * 2], y[N];
    unsigned long long state = 77;
    for (size_t i = 0; i < N; i++) {
        double x1 = test_lcg_gauss(&state);
        double z = test_lcg_gauss(&state);
        X[i * 2] = x1;
        X[i * 2 + 1] = x1 + 1e-7 * z;
        y[i] = 3.0 * X[i * 2] - 2.0 * X[i * 2 + 1] + 1.0;
    }
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 2) == MLR_OK, "init OK");
    ASSERT(mlr_linreg_fit(X, y, N, 2, 0.0, &model) == MLR_OK, "ill-conditioned fit is solvable");
    ASSERT_NEAR(model.w[0], 3.0, 1e-5, "w1 at condition number 1e8");
    ASSERT_NEAR(model.w[1], -2.0, 1e-5, "w2 at condition number 1e8");
    ASSERT_NEAR(model.b, 1.0, 1e-6, "intercept at condition number 1e8");
    mlr_lin_model_free(&model);
    PASS("linreg ill-conditioned");
}

static int test_linreg_failure_leaves_model_unchanged(void) {
    double X[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 1) == MLR_OK, "init OK");
    ASSERT(model.fitted == 0, "not fitted after init");
    double out[1];
    ASSERT(mlr_linreg_predict(X, 1, 1, &model, out) == MLR_EINVAL, "predict on an unfitted model -> EINVAL");

    ASSERT(mlr_linreg_fit(X, y, 5, 1, 0.0, &model) == MLR_OK, "good fit OK");
    ASSERT(model.fitted == 1, "fitted after a successful fit");

    // A solve that succeeds with finite weights but whose intercept (the
    // fitted value at x = 0, a long extrapolation here) overflows: the
    // failure after the solve, the last point where the model could be
    // touched. y = 1e307 - 2e305 * (x - 1002) on x = 1000..1004 keeps every
    // sum finite, gives w = -2e305, and b = 1e307 + 1002 * 2e305 is not finite.
    double X_far[] = {1000.0, 1001.0, 1002.0, 1003.0, 1004.0};
    double y_huge[] = {1.04e307, 1.02e307, 1.00e307, 0.98e307, 0.96e307};
    ASSERT(mlr_linreg_fit(X_far, y_huge, 5, 1, 0.0, &model) == MLR_EDOMAIN, "overflowing intercept -> EDOMAIN");
    ASSERT_NEAR(model.w[0], 2.0, TOL, "weights untouched by the failed fit");
    ASSERT_NEAR(model.b, 1.0, TOL, "intercept untouched by the failed fit");
    ASSERT(model.ridge == 0.0 && model.fitted == 1, "ridge and fitted untouched by the failed fit");

    // Features that overflow the design fail before the solve
    double X_huge[] = {0.0, 1e200, 2e200, 3e200, 4e200};
    ASSERT(mlr_linreg_fit(X_huge, y, 5, 1, 0.5, &model) == MLR_EDOMAIN, "overflowing design -> EDOMAIN");
    ASSERT_NEAR(model.w[0], 2.0, TOL, "weights untouched again");

    mlr_lin_model_free(&model);
    ASSERT(model.fitted == 0, "not fitted after free");
    PASS("linreg failure leaves the model unchanged");
}

static int test_linreg_underdetermined(void) {
    // n <= d with no ridge is rank deficient
    double X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double y[] = {1.0, 2.0};
    mlr_lin_model model;
    ASSERT(mlr_lin_model_init(&model, 3) == MLR_OK, "init OK");
    ASSERT(mlr_linreg_fit(X, y, 2, 3, 0.0, &model) == MLR_EDOMAIN, "n < d without ridge -> EDOMAIN");
    ASSERT(mlr_linreg_fit(X, y, 2, 3, 0.5, &model) == MLR_OK, "n < d with ridge is solvable");
    mlr_lin_model_free(&model);
    PASS("linreg underdetermined");
}

static int test_lin_model_init_free(void) {
    mlr_lin_model model;

    ASSERT(mlr_lin_model_init(&model, 5) == MLR_OK, "init OK");
    ASSERT(model.d == 5 && model.w != NULL && model.b == 0.0 && model.ridge == 0.0, "fields set");
    mlr_lin_model_free(&model);
    ASSERT(model.w == NULL && model.d == 0, "fields cleared after free");
    mlr_lin_model_free(&model);
    mlr_lin_model_free(NULL);

    ASSERT(mlr_lin_model_init(NULL, 5) == MLR_EINVAL, "NULL model -> EINVAL");
    ASSERT(mlr_lin_model_init(&model, 0) == MLR_EINVAL, "d=0 -> EINVAL");
    ASSERT(model.w == NULL && model.d == 0 && model.fitted == 0, "model left empty after EINVAL");

    // A dimension whose weight vector cannot be sized is invalid input
    ASSERT(mlr_lin_model_init(&model, SIZE_MAX / 2) == MLR_EINVAL, "unsizeable dimension -> EINVAL");
    ASSERT(model.w == NULL && model.d == 0 && model.fitted == 0, "model left empty after EINVAL");

#if SIZE_MAX <= 0xFFFFFFFFu
    // 32-bit size_t: a model that fits in memory but whose ridge-augmented
    // design (d + n rows of d doubles) does not. The size check must fire
    // before any arithmetic wraps and before the inputs are read.
    enum { BIG_D = 1 << 16 };
    static double X2[2 * BIG_D];
    double y2[] = {1.0, 2.0};
    ASSERT(mlr_lin_model_init(&model, BIG_D) == MLR_OK, "init d=2^16 OK");
    ASSERT(mlr_linreg_fit(X2, y2, 2, BIG_D, 1.0, &model) == MLR_EINVAL, "unsizeable design -> EINVAL");
    mlr_lin_model_free(&model);
#endif
    PASS("lin_model init/free");
}

int test_linreg(void) {
    int failures = 0;
    failures += test_linreg_exact_1d();
    failures += test_linreg_exact_2d();
    failures += test_linreg_ridge_on_rank_deficient();
    failures += test_linreg_scale_invariance();
    failures += test_linreg_invalid_inputs();
    failures += test_linreg_ill_conditioned();
    failures += test_linreg_failure_leaves_model_unchanged();
    failures += test_linreg_underdetermined();
    failures += test_lin_model_init_free();
    return failures;
}
