#ifndef MLRISK_LINREG_H
#define MLRISK_LINREG_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file linreg.h
 * @brief Ridge linear regression for small feature dimensions
 *
 * Least squares by Householder QR on the centered design with sqrt(ridge) I
 * appended, so accuracy is about condition number times epsilon rather
 * than its square. Centering is done on values shifted by the first row, so
 * a feature at a large level (a price near 1e9 with unit variation) is
 * centered to the precision of its variation, not its level. Intended for
 * d in the tens, not the thousands. Output arrays must not alias inputs
 * (MLR_RESTRICT, types.h).
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Linear model: y = X w + b
 *
 * The model owns w. Copying the struct by value aliases that allocation;
 * free exactly one copy. Create with mlr_lin_model_init, release with
 * mlr_lin_model_free.
 */
typedef struct {
    size_t d;        /**< Feature dimension */
    double *w;       /**< Weights (length d), owned */
    double b;        /**< Intercept */
    double ridge;    /**< Ridge parameter the model was last fitted with */
    int fitted;      /**< Exactly 1 after a successful mlr_linreg_fit, else 0 */
} mlr_lin_model;

/**
 * @brief Initialize a model, allocating its weight vector
 *
 * Call mlr_lin_model_free before re-initializing an initialized model.
 *
 * On failure the model is left empty (w == NULL, d == 0, fitted == 0) and
 * is safe to read or free.
 *
 * @param model Model to initialize
 * @param d Feature dimension (>= 1)
 * @return MLR_OK on success, MLR_EINVAL on invalid input or a dimension
 *         whose weight vector cannot be sized, MLR_ENOMEM on allocation
 *         failure
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
 * With ridge == 0 the system must be full rank, which needs n > d (centering
 * costs one rank) and linearly independent columns; otherwise MLR_EDOMAIN.
 * A ridge > 0 makes rank-deficient designs solvable, including degenerate
 * cases such as n == 1 (w = 0, b = y[0]), as long as sqrt(ridge) is not
 * itself below the rank tolerance (about 1e-15 relative to the largest
 * column). Rank deficiency is detected on the diagonal of an unpivoted R,
 * which catches exact and near-exact dependence but is not a guarantee for
 * every pathological design.
 *
 * On any failure the model is left exactly as it was.
 *
 * @param X Feature matrix, row-major: X[i*d + j] is sample i, feature j (all finite)
 * @param y Target (length n, all finite)
 * @param n Number of samples (>= 1)
 * @param d Number of features (must equal model->d)
 * @param ridge Ridge parameter (finite, >= 0); stored in model->ridge
 * @param model Model already initialized with mlr_lin_model_init (in/out)
 * @return MLR_OK on success, MLR_EINVAL on invalid input (including
 *         dimensions whose work arrays cannot be sized), MLR_ENOMEM on
 *         allocation failure, MLR_EDOMAIN on a singular or overflowing system
 */
mlr_status mlr_linreg_fit(
    const double *X,
    const double *y,
    size_t n,
    size_t d,
    double ridge,
    mlr_lin_model *model
);

/**
 * @brief Predict y = X w + b
 *
 * X is not validated for finiteness; a non-finite feature gives a non-finite
 * prediction.
 *
 * @param X Feature matrix, row-major (n rows, d columns)
 * @param n Number of samples
 * @param d Number of features (must equal model->d)
 * @param model Model with fitted == 1
 * @param out Output predictions (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input or an unfitted model
 */
mlr_status mlr_linreg_predict(
    const double *X,
    size_t n,
    size_t d,
    const mlr_lin_model *model,
    double *MLR_RESTRICT out
);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_LINREG_H */
