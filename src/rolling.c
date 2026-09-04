#include "mlrisk/rolling.h"
#include <math.h>

// Reference level for offset-shifted accumulation: the first finite value.
// Variance is shift-invariant and the mean shifts back, so subtracting a
// common level keeps precision when the series sits far from zero.
static double first_finite(const double *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (mlr_isfinite(x[i])) {
            return x[i];
        }
    }
    return 0.0;
}

mlr_status mlr_rolling_mean(const double *x, size_t n, size_t window, double *MLR_RESTRICT out) {
    if (x == NULL || out == NULL || n == 0 || window == 0) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < window - 1 && i < n; i++) {
        out[i] = MLR_NAN;
    }
    if (window > n) {
        return MLR_OK;
    }

    double offset = first_finite(x, n);

    // Sliding sum over the finite values only; `bad` counts non-finite values
    // in the current window, so the sum stays clean and output recovers as
    // soon as the last bad value leaves
    double sum = 0.0;
    size_t bad = 0;
    for (size_t j = 0; j < window; j++) {
        if (mlr_isfinite(x[j])) {
            sum += x[j] - offset;
        } else {
            bad++;
        }
    }

    for (size_t i = window - 1; i < n; i++) {
        out[i] = (bad > 0) ? MLR_NAN : offset + sum / (double)window;
        if (i + 1 < n) {
            if (mlr_isfinite(x[i + 1])) {
                sum += x[i + 1] - offset;
            } else {
                bad++;
            }
            if (mlr_isfinite(x[i + 1 - window])) {
                sum -= x[i + 1 - window] - offset;
            } else {
                bad--;
            }
        }
    }

    return MLR_OK;
}

mlr_status mlr_rolling_std(const double *x, size_t n, size_t window, double *MLR_RESTRICT out) {
    if (x == NULL || out == NULL || n == 0 || window == 0) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < window - 1 && i < n; i++) {
        out[i] = MLR_NAN;
    }
    if (window > n) {
        return MLR_OK;
    }

    if (window == 1) {
        for (size_t i = 0; i < n; i++) {
            out[i] = mlr_isfinite(x[i]) ? 0.0 : MLR_NAN;
        }
        return MLR_OK;
    }

    double w = (double)window;
    double offset = first_finite(x, n);

    size_t bad = 0;
    for (size_t j = 0; j < window; j++) {
        if (!mlr_isfinite(x[j])) {
            bad++;
        }
    }

    double mean = 0.0;
    double m2 = 0.0;
    int valid = 0;

    for (size_t i = window - 1; i < n; i++) {
        if (i >= window) {
            if (!mlr_isfinite(x[i])) {
                bad++;
            }
            if (!mlr_isfinite(x[i - window])) {
                bad--;
            }
        }

        if (bad > 0) {
            out[i] = MLR_NAN;
            valid = 0;
            continue;
        }

        if (!valid) {
            // Rebuild with standard Welford accumulation over this window
            mean = 0.0;
            m2 = 0.0;
            for (size_t j = i - window + 1; j <= i; j++) {
                double xs = x[j] - offset;
                double delta = xs - mean;
                mean += delta / (double)(j - (i - window + 1) + 1);
                m2 += delta * (xs - mean);
            }
            valid = 1;
        } else {
            // Rolling Welford: remove the oldest sample, add the newest
            double x_old = x[i - window] - offset;
            double x_new = x[i] - offset;

            double mean_removed = (w * mean - x_old) / (w - 1.0);
            m2 -= (x_old - mean) * (x_old - mean_removed);
            mean = mean_removed;

            double mean_added = mean + (x_new - mean) / w;
            m2 += (x_new - mean) * (x_new - mean_added);
            mean = mean_added;
        }

        // Removal can push m2 epsilon-negative
        out[i] = sqrt(fmax(m2, 0.0) / w);
    }

    return MLR_OK;
}

// A return usable by the EWMA recursion: finite, and its square is too
static int usable_return(double r) {
    return mlr_isfinite(r) && mlr_isfinite(r * r);
}

mlr_status mlr_ewma_vol(const double *returns, size_t n, double lambda, double *MLR_RESTRICT out) {
    if (returns == NULL || out == NULL || n == 0) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(lambda) || lambda < 0.0 || lambda >= 1.0) {
        return MLR_EINVAL;
    }

    // No forecast exists until one usable return has been seen
    size_t s = 0;
    while (s < n && !usable_return(returns[s])) {
        out[s] = MLR_NAN;
        s++;
    }
    if (s == n) {
        return MLR_OK;
    }
    out[s] = MLR_NAN;
    double variance = returns[s] * returns[s];

    // out[t] is emitted before returns[t] is absorbed: predictive alignment
    for (size_t t = s + 1; t < n; t++) {
        out[t] = sqrt(variance);
        if (usable_return(returns[t])) {
            variance = lambda * variance + (1.0 - lambda) * returns[t] * returns[t];
        }
    }

    return MLR_OK;
}
