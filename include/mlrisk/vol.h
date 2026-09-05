#ifndef MLRISK_VOL_H
#define MLRISK_VOL_H

#include "mlrisk/types.h"
#include <stddef.h>

/**
 * @file vol.h
 * @brief Volatility estimators: GARCH(1,1) and range-based (Parkinson, Garman-Klass)
 *
 * All outputs are per-period volatility (sigma). See EWMA in rolling.h.
 *
 * Timing: the GARCH filter is predictive, sigma_out[t] is the forecast for
 * period t made from returns[0..t-1]. The range estimators are per-bar and
 * contemporaneous, out[t] is measured from bar t itself, so lag them one bar
 * before using them to size a position held over bar t.
 *
 * Range estimators carry a small downward bias on discretely sampled bars
 * (the observed high/low understate the continuous extremes).
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Smallest sample mlr_garch_fit accepts */
#define MLR_GARCH_MIN_N 100
/** Feasibility bound on alpha + beta for every GARCH function */
#define MLR_GARCH_MAX_PERSISTENCE 0.9999

/**
 * @brief GARCH(1,1): sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1]
 *
 * Returns are assumed mean-zero. The recursion starts from a pre-sample
 * variance ("backcast"): sigma2[0] = omega + (alpha + beta) * backcast, the
 * same presample rule as the Python `arch` package. mlr_garch_fit uses the
 * plain mean of squared returns over the fit sample as the backcast and
 * records it in the model, so filtering later data never has to look at the
 * data being filtered. The plain mean is not outlier-robust: one bad tick in
 * the fit sample inflates the stored backcast for every later filter call.
 */
typedef struct {
    double omega;        /**< Constant term (finite, > 0) */
    double alpha;        /**< ARCH coefficient (>= 0) */
    double beta;         /**< GARCH coefficient (>= 0, alpha + beta < MLR_GARCH_MAX_PERSISTENCE) */
    double sigma2_next;  /**< One-step-ahead conditional variance after the fit sample (> 0) */
    double loglik;       /**< Maximized Gaussian log-likelihood (constants dropped) */
    int converged;       /**< Exactly 1 if the optimizer met its tolerances, exactly 0 if it
                              hit the iteration cap. Says nothing about identification. */
    double backcast;     /**< Pre-sample variance the recursion starts from; the fit
                              stores mean(r^2). 0 = the unconditional variance
                              omega / (1 - alpha - beta) */
} mlr_garch;

/**
 * @brief Fit GARCH(1,1) by Gaussian maximum likelihood
 *
 * A coarse feasible grid seeds Nelder-Mead from its three best
 * high-persistence points and its best low-persistence point, each
 * restarted from its own result until that stops improving, and the best
 * result is kept; the likelihood can have more than one local maximum.
 * Non-convergence is not an error: the best point found is returned with
 * converged == 0.
 *
 * The estimate is the maximum of the Gaussian likelihood, whatever that
 * maximum looks like. One extreme tick can make it an ARCH-like corner,
 * alpha near 1 and beta near 0, whose forecast is essentially |r[t-1]|;
 * the fitter reports that model with converged == 1 because it is the
 * answer to the question asked. Winsorise or drop the tick first if that
 * is not the model you want.
 * alpha and beta are invariant to the scale of the returns; omega scales
 * with their variance.
 *
 * Three parameters need a few hundred observations to be identified; with
 * short samples, or samples with almost no variation (a run of zeros and a
 * few nonzero returns), the optimizer converges to whatever the flat
 * likelihood allows (typically a near-unit-root model) and reports
 * converged == 1.
 *
 * @param returns Mean-zero returns (length n, all finite)
 * @param n Number of returns (must be >= MLR_GARCH_MIN_N)
 * @param model_out Fitted model
 * @return MLR_OK on success, MLR_EINVAL on invalid input, MLR_EDOMAIN if the
 *         returns have zero, non-finite, or denormally small variance
 */
mlr_status mlr_garch_fit(const double *returns, size_t n, mlr_garch *model_out);

/**
 * @brief Conditional volatility path under a fixed model
 *
 * sigma2[0] = omega + (alpha + beta) * backcast (a backcast of 0 means the
 * unconditional variance, which is a fixed point of the recursion), then
 * the GARCH recursion; sigma_out[t] = sqrt(sigma2[t]).
 *
 * The seed comes from the model, never from the returns being filtered, so
 * sigma_out[t] depends only on returns[0..t-1]: filtering a prefix of a
 * series gives exactly the prefix of the full filter, and filtering the fit
 * sample ends at sigma2_next.
 *
 * This function always starts from the backcast. To continue a fitted model
 * onto the data that follows the fit sample, either filter the fit sample
 * and the new data together, or call mlr_garch_filter_from with
 * model->sigma2_next; filtering the new data alone with this function
 * restarts the recursion from the pre-sample seed and is wrong for the
 * first few dozen periods.
 *
 * A non-finite return (missing data) does not affect sigma_out[t], which
 * was already determined; the recursion substitutes the conditional
 * expectation of r[t]^2 (the current variance) and carries on.
 *
 * @param model Fitted (or manually constructed) model
 * @param returns Mean-zero returns (length n)
 * @param n Number of returns
 * @param sigma_out Output per-period sigma (length n, pre-allocated)
 * sigma_out must not alias returns (MLR_RESTRICT, types.h).
 *
 * @return MLR_OK on success, MLR_EINVAL on invalid input or parameters
 *         (omega must be finite and > 0, alpha and beta >= 0 with
 *         alpha + beta < MLR_GARCH_MAX_PERSISTENCE, backcast finite and
 *         >= 0), MLR_EDOMAIN if the recursion overflows (extreme parameters
 *         or returns): sigma_out is then partially written and must be
 *         discarded. No output is ever Inf.
 */
mlr_status mlr_garch_filter(const mlr_garch *model, const double *returns, size_t n,
                            double *MLR_RESTRICT sigma_out);

/**
 * @brief Conditional volatility path starting from a given variance
 *
 * Same recursion as mlr_garch_filter, but sigma2[0] = sigma2_first instead of
 * the backcast rule. sigma_out[t] depends only on sigma2_first and
 * returns[0..t-1]. With sigma2_first = model->sigma2_next and returns being
 * the data that follows the fit sample, the output continues the in-sample
 * path exactly: it equals the tail of mlr_garch_filter over the fit sample
 * and the new data together, bit for bit.
 *
 * @param model Fitted (or manually constructed) model
 * @param sigma2_first Conditional variance of period 0 (finite, > 0)
 * @param returns Mean-zero returns (length n)
 * @param n Number of returns
 * @param sigma_out Output per-period sigma (length n, pre-allocated)
 * @return As mlr_garch_filter; MLR_EINVAL if sigma2_first is not finite and positive
 */
mlr_status mlr_garch_filter_from(const mlr_garch *model, double sigma2_first,
                                 const double *returns, size_t n,
                                 double *MLR_RESTRICT sigma_out);

/**
 * @brief Multi-step volatility forecast from the end of the fit sample
 *
 * sigma_out[h] is the per-period volatility h+1 steps ahead:
 * sigma2[T+1+h] = uncond + (alpha+beta)^h * (sigma2_next - uncond),
 * where uncond = omega / (1 - alpha - beta).
 *
 * @param model Fitted model (sigma2_next must be finite and > 0; a
 *              hand-built model must set it)
 * @param horizon Number of steps to forecast (>= 1)
 * @param sigma_out Output per-period sigma (length horizon, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input or parameters,
 *         MLR_EDOMAIN if the variance path overflows (sigma_out partially
 *         written)
 */
mlr_status mlr_garch_forecast(const mlr_garch *model, size_t horizon, double *MLR_RESTRICT sigma_out);

/**
 * @brief Parkinson range-based volatility, per bar
 *
 * sigma[i] = sqrt( ln(high[i]/low[i])^2 / (4 ln 2) )
 *
 * Contemporaneous: out[i] is measured from bar i. Lag it one bar before
 * sizing a position held over bar i.
 *
 * Bad bars (non-finite, <= 0, high < low, or a ratio that overflows)
 * produce out[i] = MLR_NAN; the call still returns MLR_OK.
 *
 * @param high High prices (length n)
 * @param low Low prices (length n)
 * @param n Number of bars
 * @param out Output per-bar sigma (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_parkinson_vol(const double *high, const double *low, size_t n, double *MLR_RESTRICT out);

/**
 * @brief Garman-Klass range-based volatility, per bar
 *
 * sigma2[i] = 0.5 * ln(high/low)^2 - (2 ln 2 - 1) * ln(close/open)^2
 *
 * Contemporaneous: out[i] is measured from bar i. Lag it one bar before
 * sizing a position held over bar i.
 *
 * Bad bars (non-finite, <= 0, high < low, open or close outside [low, high],
 * or a ratio that overflows) produce out[i] = MLR_NAN; the call still returns
 * MLR_OK. On a consistent bar |ln(close/open)| <= ln(high/low), so the
 * estimator is never negative.
 *
 * @param open Open prices (length n)
 * @param high High prices (length n)
 * @param low Low prices (length n)
 * @param close Close prices (length n)
 * @param n Number of bars
 * @param out Output per-bar sigma (length n, pre-allocated)
 * @return MLR_OK on success, MLR_EINVAL on invalid input
 */
mlr_status mlr_garman_klass_vol(const double *open, const double *high,
                                const double *low, const double *close,
                                size_t n, double *MLR_RESTRICT out);

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_VOL_H */
