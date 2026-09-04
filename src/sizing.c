#include "mlrisk/sizing.h"
#include <float.h>

mlr_status mlr_vol_target_position(
    const double *sigma,
    double target_vol,
    double equity,
    const double *price,
    double max_leverage,
    size_t n,
    double *position_out
) {
    if (sigma == NULL || price == NULL || position_out == NULL || n == 0) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(target_vol) || target_vol <= 0.0 ||
        !mlr_isfinite(equity) || equity <= 0.0 ||
        !mlr_isfinite(max_leverage) || max_leverage <= 0.0) {
        return MLR_EINVAL;
    }

    // All scalars are positive, so positions are >= 0 and a one-sided
    // notional cap is sufficient
    double max_notional = max_leverage * equity;

    for (size_t i = 0; i < n; i++) {
        if (!mlr_isfinite(sigma[i]) || sigma[i] <= 0.0 ||
            !mlr_isfinite(price[i]) || price[i] <= 0.0) {
            position_out[i] = 0.0;
            continue;
        }

        double position = (target_vol / sigma[i]) * (equity / price[i]);
        if (position * price[i] > max_notional) {
            position = max_notional / price[i];
        }
        // A denormal price overflows equity / price even after the cap, and a
        // denormal position has too few bits left to respect the cap; both
        // are zero
        position_out[i] = (mlr_isfinite(position) && position >= DBL_MIN) ? position : 0.0;
    }

    return MLR_OK;
}

mlr_status mlr_kelly_fraction(const double *returns, size_t n, double fraction, double *f_out) {
    if (returns == NULL || f_out == NULL) {
        return MLR_EINVAL;
    }
    if (n < 2 || !mlr_isfinite(fraction) || fraction <= 0.0) {
        return MLR_EINVAL;
    }

    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!mlr_isfinite(returns[i])) {
            return MLR_EINVAL;
        }
        mean += returns[i];
    }
    mean /= (double)n;

    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = returns[i] - mean;
        var += d * d;
    }
    var /= (double)(n - 1);

    if (!(var > 0.0)) {
        return MLR_EDOMAIN;
    }

    double f = fraction * mean / var;
    if (!mlr_isfinite(f)) {
        return MLR_EDOMAIN;
    }
    *f_out = f;
    return MLR_OK;
}

mlr_status mlr_drawdown_scale(const double *equity, size_t n, double max_dd, double *scale_out) {
    if (equity == NULL || scale_out == NULL || n == 0) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(max_dd) || max_dd <= 0.0 || max_dd > 1.0) {
        return MLR_EINVAL;
    }

    // Equity is a cumulative path: one bad value would poison the running peak
    for (size_t i = 0; i < n; i++) {
        if (!mlr_isfinite(equity[i])) {
            return MLR_EINVAL;
        }
        if (equity[i] <= 0.0) {
            return MLR_EDOMAIN;
        }
    }

    double peak = equity[0];
    for (size_t i = 0; i < n; i++) {
        if (equity[i] > peak) {
            peak = equity[i];
        }
        // equity[i] <= peak, so dd >= 0 and scale <= 1
        double dd = 1.0 - equity[i] / peak;
        double scale = 1.0 - dd / max_dd;
        scale_out[i] = scale < 0.0 ? 0.0 : scale;
    }

    return MLR_OK;
}
