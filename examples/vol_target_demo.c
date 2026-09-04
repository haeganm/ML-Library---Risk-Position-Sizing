// Walk-forward volatility targeting on a simulated GARCH series.
//
// For each walk-forward split: fit GARCH(1,1) on the training window, run the
// filter forward so every test-period sigma is a forecast made from returns
// before that period, size a position from that forecast, and score it
// against the return of the same period. If the alignment is right, the
// realized volatility of the strategy lands near the target.
//
// Usage: vol_target_demo [seed]

#include "mlrisk/mlrisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum { N = 1500, TRAIN = 500, TEST = 250 };

// Same generator as tests/test_util.h, kept local so the example depends on
// nothing but the public headers
static double lcg_u01(unsigned long long *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(*state >> 11) + 0.5) / 9007199254740992.0;
}

static double lcg_gauss(unsigned long long *state) {
    double u1 = lcg_u01(state);
    double u2 = lcg_u01(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static void simulate(unsigned long long seed, double *returns, double *prices, size_t n) {
    const double omega = 2e-6, alpha = 0.10, beta = 0.85;
    double s2 = omega / (1.0 - alpha - beta);
    prices[0] = 100.0;
    returns[0] = 0.0;
    for (size_t t = 1; t < n; t++) {
        returns[t] = sqrt(s2) * lcg_gauss(&seed);
        prices[t] = prices[t - 1] * (1.0 + returns[t]);
        s2 = omega + alpha * returns[t] * returns[t] + beta * s2;
    }
}

static double stdev(const double *x, size_t n) {
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    double var = 0.0;
    for (size_t i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    return sqrt(var / (double)n);
}

int main(int argc, char **argv) {
    unsigned long long seed = argc > 1 ? strtoull(argv[1], NULL, 10) : 42ULL;
    static double returns[N], prices[N], sigma[N], entry_price[TEST], sigma_test[TEST],
                  position[TEST], strat_ret[TEST];

    const double target_vol = 0.01;   // 1% per period
    const double equity = 100000.0;   // held constant: no compounding in this demo
    const double max_leverage = 3.0;

    simulate(seed, returns, prices, N);

    size_t count;
    if (mlr_walk_forward_splits(N, TRAIN, TEST, TEST, 0, 0, 0, NULL, 0, &count) != MLR_OK || count == 0) {
        fprintf(stderr, "no walk-forward splits fit in %d samples\n", N);
        return 1;
    }
    mlr_split *splits = malloc(count * sizeof *splits);
    if (splits == NULL || mlr_walk_forward_splits(N, TRAIN, TEST, TEST, 0, 0, 0, splits, count, &count) != MLR_OK) {
        fprintf(stderr, "split generation failed\n");
        return 1;
    }

    printf("mlrisk %s  seed %llu  target vol %.2f%%  %zu walk-forward folds\n\n",
           MLRISK_VERSION, seed, 100.0 * target_vol, count);
    printf("%4s %12s %8s %8s %6s %14s %12s\n",
           "fold", "test window", "alpha", "beta", "conv", "realized vol", "pnl");

    for (size_t k = 0; k < count; k++) {
        const mlr_split *s = &splits[k];

        mlr_garch model;
        if (mlr_garch_fit(returns + s->train_start, s->train_end - s->train_start, &model) != MLR_OK) {
            fprintf(stderr, "fold %zu: GARCH fit failed\n", k);
            return 1;
        }

        // Continue the fitted model onto the test window from the variance
        // state at the end of training: sigma[t] uses returns before t,
        // parameters from the training window only
        if (mlr_garch_filter_from(&model, model.sigma2_next, returns + s->test_start,
                                  s->test_end - s->test_start, sigma + s->test_start) != MLR_OK) {
            fprintf(stderr, "fold %zu: GARCH filter failed\n", k);
            return 1;
        }

        // A position held over period t is entered at the close of t-1
        for (size_t t = s->test_start; t < s->test_end; t++) {
            sigma_test[t - s->test_start] = sigma[t];
            entry_price[t - s->test_start] = prices[t - 1];
        }
        if (mlr_vol_target_position(sigma_test, target_vol, equity, entry_price,
                                    max_leverage, TEST, position) != MLR_OK) {
            fprintf(stderr, "fold %zu: sizing failed\n", k);
            return 1;
        }

        double pnl = 0.0;
        for (size_t t = s->test_start; t < s->test_end; t++) {
            size_t i = t - s->test_start;
            double gain = position[i] * entry_price[i] * returns[t];
            strat_ret[i] = gain / equity;
            pnl += gain;
        }

        printf("%4zu %5zu-%-6zu %8.4f %8.4f %6d %13.2f%% %12.2f\n",
               k, s->test_start, s->test_end, model.alpha, model.beta, model.converged,
               100.0 * stdev(strat_ret, TEST), pnl);
    }

    free(splits);
    return 0;
}
