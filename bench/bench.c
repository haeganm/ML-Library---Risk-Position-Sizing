// Timing of the streaming functions and the GARCH fit at increasing n. The
// point is the scaling column: an O(n) function shows a ratio near 10 when n
// grows tenfold. Usage: mlrisk_bench [repeats]
#include "mlrisk/mlrisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double lcg_gauss(unsigned long long *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = ((double)(*s >> 11) + 0.5) / 9007199254740992.0;
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = ((double)(*s >> 11) + 0.5) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

typedef void (*bench_fn)(const double *x, size_t n, double *out);
static void b_rolling_mean(const double *x, size_t n, double *out) { mlr_rolling_mean(x, n, 50, out); }
static void b_rolling_std(const double *x, size_t n, double *out) { mlr_rolling_std(x, n, 50, out); }
static void b_ewma(const double *x, size_t n, double *out) { mlr_ewma_vol(x, n, 0.94, out); }
static mlr_garch g_model;
static void b_filter(const double *x, size_t n, double *out) { mlr_garch_filter(&g_model, x, n, out); }
static void b_fit(const double *x, size_t n, double *out) { mlr_garch model; mlr_garch_fit(x, n, &model); out[0] = model.alpha; }

static double best_of(bench_fn f, const double *x, size_t n, double *out, int repeats) {
    double best = 1e300;
    for (int r = 0; r < repeats; r++) {
        double t0 = now();
        f(x, n, out);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    int repeats = argc > 1 ? atoi(argv[1]) : 5;
    const size_t NMAX = 10000000;
    double *x = malloc(NMAX * sizeof *x), *out = malloc(NMAX * sizeof *out);
    if (x == NULL || out == NULL) return 1;
    unsigned long long st = 7;
    for (size_t i = 0; i < NMAX; i++) x[i] = 0.01 * lcg_gauss(&st);
    g_model = (mlr_garch){.omega = 1e-6, .alpha = 0.1, .beta = 0.85, .converged = 1, .backcast = 0.0};

    struct { const char *name; bench_fn f; size_t sizes[3]; } table[] = {
        {"rolling_mean w=50", b_rolling_mean, {100000, 1000000, 10000000}},
        {"rolling_std  w=50", b_rolling_std, {100000, 1000000, 10000000}},
        {"ewma_vol", b_ewma, {100000, 1000000, 10000000}},
        {"garch_filter", b_filter, {100000, 1000000, 10000000}},
        {"garch_fit", b_fit, {1000, 10000, 100000}},
    };
    printf("%-20s %10s %12s %10s %8s   (best of %d)\n", "function", "n", "time", "ns/elem", "ratio", repeats);
    for (size_t k = 0; k < sizeof table / sizeof table[0]; k++) {
        double prev = 0.0;
        for (int s = 0; s < 3; s++) {
            size_t n = table[k].sizes[s];
            double t = best_of(table[k].f, x, n, out, repeats);
            if (prev > 0) {
                printf("%-20s %10zu %10.3f ms %10.1f %7.1fx\n", table[k].name, n, 1e3 * t, 1e9 * t / (double)n, t / prev);
            } else {
                printf("%-20s %10zu %10.3f ms %10.1f\n", table[k].name, n, 1e3 * t, 1e9 * t / (double)n);
            }
            prev = t;
        }
    }
    free(x); free(out);
    return 0;
}
