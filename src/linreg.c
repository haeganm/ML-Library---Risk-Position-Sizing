#include "mlrisk/linreg.h"
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

mlr_status mlr_lin_model_init(mlr_lin_model *model, size_t d) {
    if (model == NULL || d == 0) {
        return MLR_EINVAL;
    }
    if (d > SIZE_MAX / sizeof(double)) {
        model->w = NULL;
        model->d = 0;
        return MLR_ENOMEM;
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

// Least squares min ||A w - b|| by Householder QR, A is rows x d row-major
// and is overwritten; b (length rows) is overwritten with Q^T b. Solving
// the least-squares problem directly keeps the conditioning of A, where the
// normal equations would square it. Rank deficiency is detected on the
// diagonal of R relative to its largest entry.
static mlr_status qr_solve(double *A, double *b, size_t rows, size_t d, double *w) {
    double max_diag = 0.0;
    for (size_t k = 0; k < d; k++) {
        double norm2 = 0.0;
        for (size_t i = k; i < rows; i++) norm2 += A[i * d + k] * A[i * d + k];
        double norm = sqrt(norm2);
        if (norm == 0.0) {
            return MLR_EDOMAIN;
        }
        // Householder vector v = x - alpha e_k with alpha chosen to avoid cancellation
        double alpha = A[k * d + k] > 0.0 ? -norm : norm;
        double v0 = A[k * d + k] - alpha;
        double vnorm2 = v0 * v0;
        for (size_t i = k + 1; i < rows; i++) vnorm2 += A[i * d + k] * A[i * d + k];
        if (vnorm2 > 0.0) {
            // Apply H = I - 2 v v^T / (v^T v) to the remaining columns and to b
            for (size_t j = k + 1; j < d; j++) {
                double dot = v0 * A[k * d + j];
                for (size_t i = k + 1; i < rows; i++) dot += A[i * d + k] * A[i * d + j];
                double f = 2.0 * dot / vnorm2;
                A[k * d + j] -= f * v0;
                for (size_t i = k + 1; i < rows; i++) A[i * d + j] -= f * A[i * d + k];
            }
            double dot = v0 * b[k];
            for (size_t i = k + 1; i < rows; i++) dot += A[i * d + k] * b[i];
            double f = 2.0 * dot / vnorm2;
            b[k] -= f * v0;
            for (size_t i = k + 1; i < rows; i++) b[i] -= f * A[i * d + k];
        }
        A[k * d + k] = alpha;
        if (fabs(alpha) > max_diag) max_diag = fabs(alpha);
    }

    double tol = max_diag * (double)rows * DBL_EPSILON;
    for (size_t k = 0; k < d; k++) {
        if (!(fabs(A[k * d + k]) > tol)) {
            return MLR_EDOMAIN;
        }
    }

    size_t k = d;
    while (k-- > 0) {
        double s = b[k];
        for (size_t j = k + 1; j < d; j++) s -= A[k * d + j] * w[j];
        w[k] = s / A[k * d + k];
        if (!mlr_isfinite(w[k])) {
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
    // Sizes are checked before any arithmetic on them can wrap
    size_t rows = n + (ridge > 0.0 ? d : 0);
    if (n > SIZE_MAX / d || rows < n || rows > SIZE_MAX / sizeof(double) / d) {
        return MLR_ENOMEM;
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
    double *A = (double *)malloc(rows * d * sizeof(double));
    double *b = (double *)malloc(rows * sizeof(double));
    if (mu_x == NULL || A == NULL || b == NULL) {
        free(mu_x);
        free(A);
        free(b);
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

    // Centered design, so the intercept is not penalized, with sqrt(ridge) I
    // appended: min ||Xc w - yc||^2 + ridge ||w||^2
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            A[i * d + j] = X[i * d + j] - mu_x[j];
        }
        b[i] = y[i] - mu_y;
    }
    if (ridge > 0.0) {
        double sr = sqrt(ridge);
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                A[(n + i) * d + j] = (i == j) ? sr : 0.0;
            }
            b[n + i] = 0.0;
        }
    }

    mlr_status status = qr_solve(A, b, rows, d, model_out->w);
    if (status == MLR_OK) {
        double xw = 0.0;
        for (size_t j = 0; j < d; j++) {
            xw += mu_x[j] * model_out->w[j];
        }
        model_out->b = mu_y - xw;
        model_out->ridge = ridge;
        if (!mlr_isfinite(model_out->b)) {
            status = MLR_EDOMAIN;
        }
    }

    free(mu_x);
    free(A);
    free(b);
    return status;
}

mlr_status mlr_linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *MLR_RESTRICT out
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
