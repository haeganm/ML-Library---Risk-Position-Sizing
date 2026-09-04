#ifndef MLRISK_SIZING_H
#define MLRISK_SIZING_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file sizing.h
 * @brief Position sizing: volatility targeting, Kelly, drawdown scaling
 *
 * All volatility values (sigma, target_vol) are per-period.
 * Annualized to per-period: per_period = annualized / sqrt(periods_per_year).
 *
 * Timing contract: sigma[t] must be a forecast for period t made from
 * information available at the close of t-1. position_out[t] is then the
 * position held over period t and its PnL is position_out[t] * returns[t].
 * mlr_ewma_vol and mlr_garch_filter already produce forecasts aligned this
 * way; per-bar range estimators must be lagged one bar first.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Volatility-targeted position size
 *
 *   position = (target_vol / sigma) * (equity / price)
 *
 * capped so that position * price <= max_leverage * equity. A non-finite or
 * non-positive sigma or price gives a position of 0 for that index.
 *
 * To cap risk per position instead, pass the risk cap (as a fraction of
 * equity) as target_vol; the formula is identical.
 *
 * @param sigma Per-period volatility forecast (length n)
 * @param target_vol Target per-period volatility (finite, > 0)
 * @param equity Account equity (finite, > 0)
 * @param price Asset price (length n)
 * @param max_leverage Maximum notional / equity (finite, > 0)
 * @param n Length of arrays
 * @param position_out Output position sizes (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_vol_target_position(
    const double *sigma,
    double target_vol,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
);

/**
 * @brief Kelly fraction from a returns sample
 *
 *   f = fraction * mean(returns) / sample_variance(returns)
 *
 * The continuous mean-variance Kelly approximation, with the n-1 variance
 * denominator. The mean is the raw mean, not the excess over a risk-free
 * rate; subtract the funding rate from returns first if that matters. With
 * short samples mean/variance is a strongly upward-biased estimate of the
 * true edge, so use a fraction well below 1.
 *
 * f_out may be negative when the sample edge is negative; the caller decides
 * how to act on that (typically: no position).
 *
 * @param returns Per-period returns (length n, all finite)
 * @param n Number of returns (>= 2)
 * @param fraction Kelly multiplier: 1.0 = full Kelly, 0.5 = half Kelly (> 0)
 * @param f_out Receives the Kelly fraction of equity
 * @return MLR_OK on success, MLR_EINVAL on invalid input, MLR_EDOMAIN if the
 *         returns have zero variance or the estimate is not finite
 */
mlr_status mlr_kelly_fraction(const double *returns, size_t n, double fraction, double *f_out);

/**
 * @brief Drawdown-based exposure scaling
 *
 * With running peak P_i = max(equity[0..i]) and drawdown dd_i = 1 - equity[i]/P_i:
 *
 *   scale_out[i] = clamp(1 - dd_i / max_dd, 0, 1)
 *
 * Full exposure at zero drawdown, tapering linearly to zero at dd >= max_dd.
 * Multiply position sizes by scale_out.
 *
 * @param equity Cumulative equity path (length n, all finite and > 0)
 * @param n Number of samples
 * @param max_dd Drawdown at which exposure reaches zero, in (0, 1]
 * @param scale_out Output scale factors in [0, 1] (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input,
 *         MLR_EDOMAIN if any equity value is non-finite or <= 0
 */
mlr_status mlr_drawdown_scale(const double *equity, size_t n, double max_dd, double *scale_out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_SIZING_H */
