#include "mlrisk/vol.h"
#include <float.h>
#include <math.h>

#define LN2 0.69314718055994530942

/* ---------------- GARCH(1,1) ---------------- */

static int garch_params_valid(double omega, double alpha, double beta) {
    return mlr_isfinite(omega) && omega > 0.0 &&
           alpha >= 0.0 && beta >= 0.0 && alpha + beta < MLR_GARCH_MAX_PERSISTENCE;
}

// The one recursion step, used by the likelihood, the fit and the filters.
// A single definition matters: alpha * r * r and alpha * (r * r) differ by an
// ulp often enough that sigma2_next would not match the filter's own path.
static double garch_step(double omega, double alpha, double beta, double r2, double s2) {
    return omega + alpha * r2 + beta * s2;
}

// First conditional variance from the pre-sample variance: the recursion
// applied as if both r[-1]^2 and sigma2[-1] equalled the backcast.
static double garch_seed(double omega, double alpha, double beta, double backcast) {
    return omega + (alpha + beta) * backcast;
}

// Gaussian negative log-likelihood (constants dropped). Infeasible parameters
// return HUGE_VAL, which is how the optimizer sees the constraints.
static double garch_nll(const double *r, size_t n, double backcast,
                        double omega, double alpha, double beta) {
    if (!garch_params_valid(omega, alpha, beta)) {
        return HUGE_VAL;
    }

    double s2 = garch_seed(omega, alpha, beta, backcast);
    double nll = 0.0;
    for (size_t t = 0; t < n; t++) {
        double r2 = r[t] * r[t];
        nll += log(s2) + r2 / s2;
        s2 = garch_step(omega, alpha, beta, r2, s2);
    }
    return 0.5 * nll;
}

typedef struct {
    const double *r;
    size_t n;
    double backcast;
} garch_ctx;

static double nm_eval(const garch_ctx *ctx, const double x[3]) {
    return garch_nll(ctx->r, ctx->n, ctx->backcast, x[0], x[1], x[2]);
}

// Nelder-Mead in 3 dimensions, standard coefficients (reflection 1,
// expansion 2, contraction 0.5, shrink 0.5). Converges when the simplex is
// small AND the function values agree; the function-value test alone is not
// enough because the objective carries an additive constant that depends on
// the units of the returns. Simplex size is measured against a fixed scale
// per parameter (the backcast for omega, 1 for alpha and beta) rather than
// the parameter's own magnitude, so a maximum on the boundary alpha = 0 (no
// ARCH effect) converges instead of running to the iteration cap.
// Returns 1 if converged, 0 if it hit the iteration cap.
static int nelder_mead3(const garch_ctx *ctx, double x[3], double *f_out) {
    enum { DIM = 3, PTS = 4, MAX_ITER = 2000 };
    const double X_TOL = 1e-10;
    const double F_TOL = 1e-12;
    const double scale[DIM] = {ctx->backcast, 1.0, 1.0};
    double simplex[PTS][DIM];
    double f[PTS];

    for (int i = 0; i < PTS; i++) {
        for (int j = 0; j < DIM; j++) {
            simplex[i][j] = x[j];
        }
        if (i > 0) {
            int j = i - 1;
            simplex[i][j] += (simplex[i][j] != 0.0) ? 0.1 * simplex[i][j] : 1e-6;
        }
        f[i] = nm_eval(ctx, simplex[i]);
    }

    int converged = 0;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Insertion sort by f, best first
        for (int i = 1; i < PTS; i++) {
            double fi = f[i];
            double xi[DIM];
            for (int j = 0; j < DIM; j++) xi[j] = simplex[i][j];
            int k = i - 1;
            while (k >= 0 && f[k] > fi) {
                f[k + 1] = f[k];
                for (int j = 0; j < DIM; j++) simplex[k + 1][j] = simplex[k][j];
                k--;
            }
            f[k + 1] = fi;
            for (int j = 0; j < DIM; j++) simplex[k + 1][j] = xi[j];
        }

        double diameter = 0.0;
        for (int i = 1; i < PTS; i++) {
            for (int j = 0; j < DIM; j++) {
                double rel = fabs(simplex[i][j] - simplex[0][j]) / scale[j];
                if (rel > diameter) diameter = rel;
            }
        }
        if (diameter <= X_TOL && f[PTS - 1] - f[0] <= F_TOL * (fabs(f[0]) + 1.0)) {
            converged = 1;
            break;
        }

        double c[DIM];
        for (int j = 0; j < DIM; j++) {
            c[j] = (simplex[0][j] + simplex[1][j] + simplex[2][j]) / 3.0;
        }

        double xr[DIM], xe[DIM], xc[DIM];
        for (int j = 0; j < DIM; j++) xr[j] = c[j] + (c[j] - simplex[PTS - 1][j]);
        double fr = nm_eval(ctx, xr);

        if (fr < f[0]) {
            for (int j = 0; j < DIM; j++) xe[j] = c[j] + 2.0 * (c[j] - simplex[PTS - 1][j]);
            double fe = nm_eval(ctx, xe);
            if (fe < fr) {
                for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xe[j];
                f[PTS - 1] = fe;
            } else {
                for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xr[j];
                f[PTS - 1] = fr;
            }
        } else if (fr < f[PTS - 2]) {
            for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xr[j];
            f[PTS - 1] = fr;
        } else {
            // Outside contraction if the reflection beat the worst point, else inside
            double sign = (fr < f[PTS - 1]) ? 0.5 : -0.5;
            for (int j = 0; j < DIM; j++) xc[j] = c[j] + sign * (c[j] - simplex[PTS - 1][j]);
            double fc = nm_eval(ctx, xc);
            double fbar = (fr < f[PTS - 1]) ? fr : f[PTS - 1];
            if (fc <= fbar) {
                for (int j = 0; j < DIM; j++) simplex[PTS - 1][j] = xc[j];
                f[PTS - 1] = fc;
            } else {
                for (int i = 1; i < PTS; i++) {
                    for (int j = 0; j < DIM; j++) {
                        simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                    }
                    f[i] = nm_eval(ctx, simplex[i]);
                }
            }
        }
    }

    // The simplex is only sorted at the top of an iteration; a shrink can
    // leave a better point elsewhere
    int best = 0;
    for (int i = 1; i < PTS; i++) {
        if (f[i] < f[best]) best = i;
    }
    for (int j = 0; j < DIM; j++) x[j] = simplex[best][j];
    *f_out = f[best];
    return converged;
}

mlr_status mlr_garch_fit(const double *returns, size_t n, mlr_garch *model_out) {
    if (returns == NULL || model_out == NULL) {
        return MLR_EINVAL;
    }
    if (n < MLR_GARCH_MIN_N) {
        return MLR_EINVAL;
    }

    double backcast = 0.0;
    for (size_t t = 0; t < n; t++) {
        if (!mlr_isfinite(returns[t])) {
            return MLR_EINVAL;
        }
        backcast += returns[t] * returns[t];
    }
    backcast /= (double)n;
    // A denormal variance would run the whole likelihood in the denormal
    // range, where the estimates degrade; the header promises EDOMAIN
    if (!mlr_isfinite(backcast) || backcast < DBL_MIN) {
        return MLR_EDOMAIN;
    }

    garch_ctx ctx = {returns, n, backcast};

    // Coarse feasible grid; omega from variance targeting so every point has
    // unconditional variance equal to the backcast. Betas below LOW_BETA
    // form a second group of seeds, used for one extra start (see below).
    static const double alphas[] = {0.02, 0.05, 0.10, 0.15};
    static const double betas[] = {0.0, 0.5, 0.80, 0.88, 0.94};
    enum { GRID = 20, HIGH_STARTS = 3, LOW_STARTS = 1, RESTARTS = 3 };
    static const double LOW_BETA = 0.8;
    _Static_assert(GRID == (int)(sizeof alphas / sizeof alphas[0]) * (int)(sizeof betas / sizeof betas[0]),
                   "GRID must equal the number of seed points");
    double grid_x[GRID][3];
    double grid_f[GRID];
    int g = 0;
    for (size_t a = 0; a < sizeof alphas / sizeof alphas[0]; a++) {
        for (size_t b = 0; b < sizeof betas / sizeof betas[0]; b++) {
            grid_x[g][0] = backcast * (1.0 - alphas[a] - betas[b]);
            grid_x[g][1] = alphas[a];
            grid_x[g][2] = betas[b];
            grid_f[g] = nm_eval(&ctx, grid_x[g]);
            g++;
        }
    }

    // Nelder-Mead from the best HIGH_STARTS high-persistence grid points and
    // the best LOW_STARTS low-persistence one, keeping the best result. The
    // likelihood can have more than one local maximum (a tiny ARCH effect
    // with high persistence, or a variance regime change, both give a second
    // basin), and a single start from the best grid point can land in the
    // wrong one. The low-persistence start covers short samples with no
    // ARCH effect, whose maximum can sit at beta = 0: from any seed with
    // beta >= 0.8 the optimizer settles in the flat near-unit-root basin at
    // alpha = 0 instead, 0.2 log-likelihood units short, and reports
    // convergence. The two groups are kept separate so that the
    // high-persistence starts always run: on a series with a large outlier
    // the low seeds score best on the grid and a best-three selection
    // would take only them, losing the high-persistence optimum on the
    // series where that one is better. Where the corner solution (alpha
    // near 1, beta near 0) genuinely has the higher likelihood, as it does
    // when one tick dominates the sample, it wins and is reported; that is
    // the maximum-likelihood estimate and the header says so. Each start is
    // re-run from its own result with a fresh simplex until that stops
    // helping, which is what gets Nelder-Mead moving again after it stalls
    // against the persistence bound.
    double best[3] = {0.0, 0.0, 0.0};
    double f_min = HUGE_VAL;
    int converged = 0;
    for (int k = 0; k < HIGH_STARTS + LOW_STARTS; k++) {
        const int want_low = k >= HIGH_STARTS;
        int bi = -1;
        for (int i = 0; i < GRID; i++) {
            if ((grid_x[i][2] < LOW_BETA) != want_low) continue;
            if (grid_f[i] < HUGE_VAL && (bi < 0 || grid_f[i] < grid_f[bi])) bi = i;
        }
        if (bi < 0) continue;
        grid_f[bi] = HUGE_VAL;

        double x[3] = {grid_x[bi][0], grid_x[bi][1], grid_x[bi][2]};
        double f;
        int c = nelder_mead3(&ctx, x, &f);
        for (int r = 0; r < RESTARTS; r++) {
            double x2[3] = {x[0], x[1], x[2]};
            double f2;
            int c2 = nelder_mead3(&ctx, x2, &f2);
            if (!(f2 < f - 1e-9 * (fabs(f) + 1.0))) break;
            x[0] = x2[0]; x[1] = x2[1]; x[2] = x2[2];
            f = f2;
            c = c2;
        }
        if (f < f_min) {
            f_min = f;
            best[0] = x[0]; best[1] = x[1]; best[2] = x[2];
            converged = c;
        }
    }
    if (f_min == HUGE_VAL) {
        return MLR_EDOMAIN;
    }

    model_out->omega = best[0];
    model_out->alpha = best[1];
    model_out->beta = best[2];
    model_out->loglik = -f_min;
    model_out->converged = converged;
    model_out->backcast = backcast;

    double s2 = garch_seed(best[0], best[1], best[2], backcast);
    for (size_t t = 0; t < n; t++) {
        s2 = garch_step(best[0], best[1], best[2], returns[t] * returns[t], s2);
    }
    model_out->sigma2_next = s2;

    return MLR_OK;
}

static mlr_status garch_run(const mlr_garch *model, double s2, const double *returns, size_t n,
                            double *MLR_RESTRICT sigma_out) {
    for (size_t t = 0; t < n; t++) {
        // Extreme parameters or returns can overflow the recursion; fail
        // rather than emit Inf
        if (!mlr_isfinite(s2)) {
            return MLR_EDOMAIN;
        }
        sigma_out[t] = sqrt(s2);
        // A missing observation (or one whose square overflows) is replaced
        // by its conditional expectation E[r^2 | past] = s2
        double r2 = returns[t] * returns[t];
        if (!mlr_isfinite(r2)) {
            r2 = s2;
        }
        s2 = garch_step(model->omega, model->alpha, model->beta, r2, s2);
    }

    return MLR_OK;
}

mlr_status mlr_garch_filter(const mlr_garch *model, const double *returns, size_t n,
                            double *MLR_RESTRICT sigma_out) {
    if (model == NULL || returns == NULL || sigma_out == NULL || n == 0) {
        return MLR_EINVAL;
    }
    if (!garch_params_valid(model->omega, model->alpha, model->beta)) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(model->backcast) || model->backcast < 0.0) {
        return MLR_EINVAL;
    }

    double backcast = model->backcast > 0.0
                          ? model->backcast
                          : model->omega / (1.0 - model->alpha - model->beta);
    double s2 = garch_seed(model->omega, model->alpha, model->beta, backcast);
    return garch_run(model, s2, returns, n, sigma_out);
}

mlr_status mlr_garch_filter_from(const mlr_garch *model, double sigma2_first,
                                 const double *returns, size_t n,
                                 double *MLR_RESTRICT sigma_out) {
    if (model == NULL || returns == NULL || sigma_out == NULL || n == 0) {
        return MLR_EINVAL;
    }
    if (!garch_params_valid(model->omega, model->alpha, model->beta)) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(sigma2_first) || sigma2_first <= 0.0) {
        return MLR_EINVAL;
    }
    return garch_run(model, sigma2_first, returns, n, sigma_out);
}

mlr_status mlr_garch_forecast(const mlr_garch *model, size_t horizon, double *MLR_RESTRICT sigma_out) {
    if (model == NULL || sigma_out == NULL || horizon == 0) {
        return MLR_EINVAL;
    }
    if (!garch_params_valid(model->omega, model->alpha, model->beta)) {
        return MLR_EINVAL;
    }
    if (!mlr_isfinite(model->sigma2_next) || model->sigma2_next <= 0.0) {
        return MLR_EINVAL;
    }

    double persistence = model->alpha + model->beta;
    double uncond = model->omega / (1.0 - persistence);
    if (!mlr_isfinite(uncond)) {
        return MLR_EDOMAIN;
    }
    double decay = 1.0;

    for (size_t h = 0; h < horizon; h++) {
        double s2 = uncond + decay * (model->sigma2_next - uncond);
        if (!mlr_isfinite(s2)) {
            return MLR_EDOMAIN;
        }
        sigma_out[h] = sqrt(s2);
        decay *= persistence;
    }

    return MLR_OK;
}

/* ---------------- Range-based estimators ---------------- */

static int bad_hl_bar(double high, double low) {
    return !mlr_isfinite(high) || !mlr_isfinite(low) ||
           high <= 0.0 || low <= 0.0 || high < low;
}

static double finite_or_nan(double x) {
    return mlr_isfinite(x) ? x : MLR_NAN;
}

mlr_status mlr_parkinson_vol(const double *high, const double *low, size_t n, double *MLR_RESTRICT out) {
    if (high == NULL || low == NULL || out == NULL || n == 0) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < n; i++) {
        if (bad_hl_bar(high[i], low[i])) {
            out[i] = MLR_NAN;
            continue;
        }
        double hl = log(high[i] / low[i]);
        out[i] = finite_or_nan(sqrt(hl * hl / (4.0 * LN2)));
    }

    return MLR_OK;
}

mlr_status mlr_garman_klass_vol(const double *open, const double *high,
                                const double *low, const double *close,
                                size_t n, double *MLR_RESTRICT out) {
    if (open == NULL || high == NULL || low == NULL || close == NULL || out == NULL || n == 0) {
        return MLR_EINVAL;
    }

    for (size_t i = 0; i < n; i++) {
        if (bad_hl_bar(high[i], low[i]) ||
            !mlr_isfinite(open[i]) || open[i] < low[i] || open[i] > high[i] ||
            !mlr_isfinite(close[i]) || close[i] < low[i] || close[i] > high[i]) {
            out[i] = MLR_NAN;
            continue;
        }
        // open and close lie in [low, high], so |co| <= hl and
        // sigma2 >= (0.5 - (2 ln 2 - 1)) * hl^2 >= 0
        double hl = log(high[i] / low[i]);
        double co = log(close[i] / open[i]);
        double sigma2 = 0.5 * hl * hl - (2.0 * LN2 - 1.0) * co * co;
        out[i] = finite_or_nan(sqrt(sigma2));
    }

    return MLR_OK;
}
