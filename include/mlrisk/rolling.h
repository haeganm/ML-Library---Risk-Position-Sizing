#ifndef MLRISK_ROLLING_H
#define MLRISK_ROLLING_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file rolling.h
 * @brief Rolling statistics and EWMA volatility
 *
 * Output arrays must not alias inputs (see MLR_RESTRICT in types.h).
 *
 * Trailing windows: out[i] is computed from x[i-window+1..i], which includes
 * x[i]. A rolling statistic at index t therefore knows period t; lag it one
 * bar before using it to size a position held over period t (mlr_ewma_vol
 * below is already aligned that way). Indices without a full window are
 * MLR_NAN, and window > n gives all MLR_NAN with MLR_OK. A non-finite input
 * (missing data) makes every window containing it MLR_NAN; output recovers
 * once the value leaves the window. Population variance throughout (divides
 * by window; pandas rolling().std() defaults to the sample convention).
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
mlr_status mlr_rolling_mean(const double *x, size_t n, size_t window, double *MLR_RESTRICT out);

/**
 * @brief Rolling standard deviation, O(n) on clean data
 *
 * Rolling Welford updates on offset-shifted values, so precision does not
 * degrade at large price levels. After each gap (a window containing a
 * non-finite value) the accumulators are rebuilt in O(window).
 *
 * @param x Input (length n)
 * @param n Length of x
 * @param window Window size (>= 1)
 * @param out Output (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_rolling_std(const double *x, size_t n, size_t window, double *MLR_RESTRICT out);

/**
 * @brief EWMA volatility forecast (RiskMetrics)
 *
 * Predictive: out[t] is the forecast for period t made from returns[0..t-1],
 *
 *   sigma2[t] = lambda * sigma2[t-1] + (1 - lambda) * returns[t-1]^2
 *
 * so position sizes computed from out[t] can be applied to returns[t]
 * without lookahead. The first usable return r[s] (finite, with a finite
 * square) seeds the variance: out[0..s] are MLR_NAN and out[s+1] = |r[s]|.
 * If no usable return precedes the last index (n == 1, say) every output is
 * MLR_NAN and the call still returns MLR_OK. That seed is a single
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
mlr_status mlr_ewma_vol(const double *returns, size_t n, double lambda, double *MLR_RESTRICT out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_ROLLING_H */
