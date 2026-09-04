#include "mlrisk/linreg.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

mlr_status mlr_lin_model_init(mlr_lin_model *model, size_t d) {
    if (model == NULL || d == 0) {
        return MLR_EINVAL;
    }

    model->w = (double *)calloc(d, sizeof(double));
    if (model->w == NULL) {
        model->d = 0;
        return MLR_ENOMEM;
    }
    model->d = d;
    model->b = 0.0;
    model->ridge = 0.0;
    return MLR_OK;
}

void mlr_lin_model_free(mlr_lin_model *model) {
    if (model != NULL) {
        free(model->w);
        model->w = NULL;
        model->d = 0;
    }
}

// Solve A x = b (A is d x d row-major) by Gaussian elimination with partial
// pivoting. The singularity threshold is relative to the largest entry of
// A, so a well-conditioned system at any scale is accepted.
static mlr_status solve_linear_system(const double *A, const double *b, size_t d, double *x) {
    double *aug = (double *)malloc(d * (d + 1) * sizeof(double));
    if (aug == NULL) {
        return MLR_ENOMEM;
    }

    double max_a = 0.0;
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            aug[i * (d + 1) + j] = A[i * d + j];
            if (fabs(A[i * d + j]) > max_a) {
                max_a = fabs(A[i * d + j]);
            }
        }
        aug[i * (d + 1) + d] = b[i];
    }
    // Finite inputs can still overflow the normal matrix
    if (max_a == 0.0 || !mlr_isfinite(max_a)) {
        free(aug);
        return MLR_EDOMAIN;
    }
    double singular_tol = max_a * (double)d * DBL_EPSILON;

    for (size_t col = 0; col < d; col++) {
        size_t max_row = col;
        double max_val = fabs(aug[col * (d + 1) + col]);
        for (size_t row = col + 1; row < d; row++) {
            double val = fabs(aug[row * (d + 1) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Written so that a NaN pivot is rejected, not accepted (inputs are
        // validated finite, so this is defensive)
        if (!(max_val >= singular_tol)) {
            free(aug);
            return MLR_EDOMAIN;
        }

        if (max_row != col) {
            for (size_t j = 0; j < d + 1; j++) {
                double temp = aug[col * (d + 1) + j];
                aug[col * (d + 1) + j] = aug[max_row * (d + 1) + j];
                aug[max_row * (d + 1) + j] = temp;
            }
        }

        for (size_t row = col + 1; row < d; row++) {
            double factor = aug[row * (d + 1) + col] / aug[col * (d + 1) + col];
            for (size_t j = col; j < d + 1; j++) {
                aug[row * (d + 1) + j] -= factor * aug[col * (d + 1) + j];
            }
        }
    }

    size_t i = d;
    while (i-- > 0) {
        x[i] = aug[i * (d + 1) + d];
        for (size_t j = i + 1; j < d; j++) {
            x[i] -= aug[i * (d + 1) + j] * x[j];
        }
        x[i] /= aug[i * (d + 1) + i];
    }

    free(aug);
    for (i = 0; i < d; i++) {
        if (!mlr_isfinite(x[i])) {
            return MLR_EDOMAIN;
        }
    }
    return MLR_OK;
}

mlr_status mlr_linreg_fit(
    const double *X,
    const double *y,
    size_t n,
    size_t d,
    double ridge,
    mlr_lin_model *model_out
) {
    if (X == NULL || y == NULL || model_out == NULL || n == 0 || d == 0) {
        return MLR_EINVAL;
    }
    if (model_out->d != d || model_out->w == NULL) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(ridge) || ridge < 0.0) {
        return MLR_EINVAL;
    }
    for (size_t i = 0; i < n; i++) {
        if (!mlr_isfinite(y[i])) {
            return MLR_EINVAL;
        }
        for (size_t j = 0; j < d; j++) {
            if (!mlr_isfinite(X[i * d + j])) {
                return MLR_EINVAL;
            }
        }
    }

    double *mu_x = (double *)calloc(d, sizeof(double));
    double *XtX = (double *)calloc(d * d, sizeof(double));
    double *Xty = (double *)calloc(d, sizeof(double));
    if (mu_x == NULL || XtX == NULL || Xty == NULL) {
        free(mu_x);
        free(XtX);
        free(Xty);
        return MLR_ENOMEM;
    }

    double mu_y = 0.0;
    for (size_t i = 0; i < n; i++) {
        mu_y += y[i];
        for (size_t j = 0; j < d; j++) {
            mu_x[j] += X[i * d + j];
        }
    }
    mu_y /= (double)n;
    for (size_t j = 0; j < d; j++) {
        mu_x[j] /= (double)n;
    }

    // Centered normal equations; the ridge term does not touch the intercept
    for (size_t k = 0; k < n; k++) {
        double yc = y[k] - mu_y;
        for (size_t i = 0; i < d; i++) {
            double xc_i = X[k * d + i] - mu_x[i];
            Xty[i] += xc_i * yc;
            for (size_t j = i; j < d; j++) {
                XtX[i * d + j] += xc_i * (X[k * d + j] - mu_x[j]);
            }
        }
    }
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < i; j++) {
            XtX[i * d + j] = XtX[j * d + i];
        }
        XtX[i * d + i] += ridge;
    }

    mlr_status status = solve_linear_system(XtX, Xty, d, model_out->w);
    if (status == MLR_OK) {
        double xw = 0.0;
        for (size_t j = 0; j < d; j++) {
            xw += mu_x[j] * model_out->w[j];
        }
        model_out->b = mu_y - xw;
        model_out->ridge = ridge;
    }

    free(XtX);
    free(Xty);
    free(mu_x);
    return status;
}

mlr_status mlr_linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *out
) {
    if (X == NULL || model == NULL || out == NULL || n == 0 || d == 0) {
        return MLR_EINVAL;
    }
    if (model->d != d || model->w == NULL) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < n; i++) {
        double pred = model->b;
        for (size_t j = 0; j < d; j++) {
            pred += X[i * d + j] * model->w[j];
        }
        out[i] = pred;
    }

    return MLR_OK;
}
