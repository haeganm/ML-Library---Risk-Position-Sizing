#ifndef MLRISK_ROLLING_H
#define MLRISK_ROLLING_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file rolling.h
 * @brief Rolling statistics and EWMA volatility
 *
 * Trailing windows: out[i] is computed from x[i-window+1..i]. Indices without
 * a full window are MLR_NAN. A non-finite input (missing data) makes every
 * window containing it MLR_NAN; output recovers once the value leaves the
 * window. Population variance throughout (divides by window; pandas
 * rolling().std() defaults to the sample convention).
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Rolling mean, O(n)
 *
 * @param x Input (length n)
 * @param n Length of x
 * @param window Window size (>= 1)
 * @param out Output (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_rolling_mean(const double *x, size_t n, size_t window, double *out);

/**
 * @brief Rolling standard deviation, O(n) amortized
 *
 * Rolling Welford updates on offset-shifted values, so precision does not
 * degrade at large price levels. After a window containing missing data
 * the accumulators are rebuilt once in O(window).
 *
 * @param x Input (length n)
 * @param n Length of x
 * @param window Window size (>= 1)
 * @param out Output (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_rolling_std(const double *x, size_t n, size_t window, double *out);

/**
 * @brief EWMA volatility forecast (RiskMetrics)
 *
 * Predictive: out[t] is the forecast for period t made from returns[0..t-1],
 *
 *   sigma2[t] = lambda * sigma2[t-1] + (1 - lambda) * returns[t-1]^2
 *
 * so position sizes computed from out[t] can be applied to returns[t]
 * without lookahead. The first finite return r[s] seeds the variance:
 * out[0..s] are MLR_NAN and out[s+1] = |r[s]|. That seed is a single
 * observation; at lambda 0.94 its weight decays below 5% after about 50
 * periods, so treat the start of the series as warmup.
 *
 * A non-finite return (missing data), or one whose square overflows, does
 * not change out[t], which was already determined; it is skipped and the
 * recursion state carries forward unchanged.
 *
 * @param returns Returns (length n)
 * @param n Length of returns
 * @param lambda Decay factor, finite, in [0, 1). Typical daily values 0.94-0.97.
 * @param out Output per-period sigma (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_ewma_vol(const double *returns, size_t n, double lambda, double *out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_ROLLING_H */
