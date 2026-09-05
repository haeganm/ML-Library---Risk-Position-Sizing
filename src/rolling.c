#include "mlrisk/rolling.h"
#include <math.h>

// Sliding-window statistics are accumulated on values shifted by an
// offset taken from inside the current window, and the accumulators are
// rebuilt from scratch every `window` steps (amortized O(1) per element)
// so the offset can never drift far from the data. A single global offset
// would make every later window inaccurate after a bad first tick, and a
// long trend away from the starting level would erode precision the same
// way. Windows containing a non-finite value emit MLR_NAN.

// The offset is the window's last finite value: rebuilds happen every
// `window` steps, so that value stays inside every window until the next
// rebuild and |x - offset| never exceeds the window's own range
static double window_offset(const double *x, size_t start, size_t window) {
    size_t j = start + window;
    while (j-- > start) {
        if (mlr_isfinite(x[j])) {
            return x[j];
        }
    }
    return 0.0;
}

static size_t count_bad(const double *x, size_t start, size_t window) {
    size_t bad = 0;
    for (size_t j = start; j < start + window; j++) {
        if (!mlr_isfinite(x[j])) {
            bad++;
        }
    }
    return bad;
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

    double offset = 0.0;
    double sum = 0.0;
    size_t bad = 0;
    int overflowed = 0;

    for (size_t i = window - 1; i < n; i++) {
        size_t start = i - window + 1;
        int rebuild = (start % window == 0) || overflowed;
        if (!rebuild) {
            // Slide: x[i] enters, x[i - window] leaves; the sum only ever
            // holds finite values, so it is clean the moment `bad` returns to 0
            if (mlr_isfinite(x[i])) {
                sum += x[i] - offset;
            } else {
                bad++;
            }
            if (mlr_isfinite(x[i - window])) {
                double leaving = x[i - window] - offset;
                sum -= leaving;
                // A leaving value that dwarfs what remains (an outlier) has
                // left its rounding behind in the sum; rebuild instead
                if (fabs(leaving) > 1e6 * fabs(sum)) {
                    rebuild = 1;
                }
            } else {
                bad--;
            }
        }
        if (rebuild) {
            offset = window_offset(x, start, window);
            sum = 0.0;
            bad = 0;
            for (size_t j = start; j <= i; j++) {
                if (mlr_isfinite(x[j])) {
                    sum += x[j] - offset;
                } else {
                    bad++;
                }
            }
        }
        // Finite inputs whose differences exceed DBL_MAX overflow the shifted
        // sum; that window is NaN and the state is rebuilt on the next step
        overflowed = !mlr_isfinite(sum);
        double mean = offset + sum / (double)window;
        out[i] = (bad > 0 || !mlr_isfinite(mean)) ? MLR_NAN : mean;
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
    double offset = 0.0;
    double mean = 0.0;
    double m2 = 0.0;
    size_t bad = 0;
    int valid = 0;

    for (size_t i = window - 1; i < n; i++) {
        size_t start = i - window + 1;
        int periodic = (start % window == 0);

        if (periodic) {
            bad = count_bad(x, start, window);
        } else {
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

        int rebuild = periodic || !valid;
        if (!rebuild) {
            // Rolling Welford: remove the oldest sample, add the newest
            double x_old = x[i - window] - offset;
            double x_new = x[i] - offset;
            double m2_before = m2;

            double mean_removed = (w * mean - x_old) / (w - 1.0);
            m2 -= (x_old - mean) * (x_old - mean_removed);
            mean = mean_removed;

            // If the sample that left carried most of the variance (an
            // outlier leaving the window) the subtraction above has
            // cancelled; rebuild instead of trusting it. The threshold is
            // deliberately loose: a spike that took the variance down by a
            // factor of 1e6 rather than 1e7 used to slip past a 1e-6 cliff
            // and leave eight lost digits until the next periodic rebuild.
            // A drop this large from one sample is rare, so the amortized
            // cost is unchanged.
            if (m2 < 0.25 * m2_before) {
                rebuild = 1;
            } else {
                double mean_added = mean + (x_new - mean) / w;
                m2 += (x_new - mean) * (x_new - mean_added);
                mean = mean_added;
            }
        }
        if (rebuild) {
            offset = window_offset(x, start, window);
            mean = 0.0;
            m2 = 0.0;
            for (size_t j = start; j <= i; j++) {
                double xs = x[j] - offset;
                double delta = xs - mean;
                mean += delta / (double)(j - start + 1);
                m2 += delta * (xs - mean);
            }
            valid = 1;
        }

        // Finite inputs whose differences exceed DBL_MAX overflow the shifted
        // accumulators; that window is NaN and the state is rebuilt next step
        if (!mlr_isfinite(mean) || !mlr_isfinite(m2)) {
            out[i] = MLR_NAN;
            valid = 0;
            continue;
        }
        // Removal can push m2 epsilon-negative
        out[i] = sqrt((m2 > 0.0 ? m2 : 0.0) / w);
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
