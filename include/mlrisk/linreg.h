#ifndef MLRISK_LINREG_H
#define MLRISK_LINREG_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file linreg.h
 * @brief Ridge linear regression for small feature dimensions
 *
 * Closed-form ridge regression via the normal equations, solved with
 * Gaussian elimination with partial pivoting. Intended for d in the tens,
 * not the thousands.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Linear model: y = X w + b
 */
typedef struct {
    size_t d;        /**< Feature dimension */
    double *w;       /**< Weights (length d) */
    double b;        /**< Intercept */
    double ridge;    /**< Ridge parameter the model was last fitted with */
} mlr_lin_model;

/**
 * @brief Initialize a model, allocating its weight vector
 *
 * Call mlr_lin_model_free before re-initializing an initialized model.
 *
 * @param model Model to initialize
 * @param d Feature dimension (>= 1)
 * @return MLR_OK on success, MLR_EINVAL on invalid input, MLR_ENOMEM on allocation failure
 */
mlr_status mlr_lin_model_init(mlr_lin_model *model, size_t d);

/**
 * @brief Release a model's weight vector. Safe on NULL and on a freed model.
 */
void mlr_lin_model_free(mlr_lin_model *model);

/**
 * @brief Fit by ridge regression
 *
 * Features and target are centered, so the intercept is not penalized:
 *
 *   (Xc^T Xc + ridge * I) w = Xc^T yc,   b = mean(y) - mean(X) . w
 *
 * With ridge == 0 the system must be full rank, which needs n > d and
 * linearly independent columns; otherwise MLR_EDOMAIN. With ridge > 0 the
 * system is always solvable. Note that n <= d with ridge == 0 is an exactly
 * determined fit with zero residual, which is never what you want.
 *
 * @param X Feature matrix, row-major: X[i*d + j] is sample i, feature j (all finite)
 * @param y Target (length n, all finite)
 * @param n Number of samples (>= 1)
 * @param d Number of features (must equal model_out->d)
 * @param ridge Ridge parameter (finite, >= 0); stored in model_out->ridge
 * @param model_out Model initialized with mlr_lin_model_init
 * @return MLR_OK on success, MLR_EINVAL on invalid input, MLR_ENOMEM on
 *         allocation failure, MLR_EDOMAIN on a singular system
 */
mlr_status mlr_linreg_fit(
    const double *X,
    const double *y,
    size_t n,
    size_t d,
    double ridge,
    mlr_lin_model *model_out
);

/**
 * @brief Predict y = X w + b
 *
 * @param X Feature matrix, row-major (n rows, d columns)
 * @param n Number of samples
 * @param d Number of features (must equal model->d)
 * @param model Fitted model
 * @param out Output predictions (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *out
);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_LINREG_H */
