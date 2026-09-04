#ifndef MLRISK_TEST_UTIL_H
#define MLRISK_TEST_UTIL_H

#include <stdio.h>
#include <math.h>

// A failed assertion ends the enclosing test function; later assertions in
// the same function are not evaluated.
#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabs((double)(a) - (double)(b)) <= (tol), msg)

#define PASS(name) do { printf("  PASS: %s\n", name); return 0; } while (0)

// Deterministic 64-bit LCG so every platform sees identical samples.
// tests/reference/garch_arch_reference.py mirrors these two functions
// operation for operation; keep them in sync.
static inline double test_lcg_u01(unsigned long long *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(*state >> 11) + 0.5) / 9007199254740992.0;
}

// Standard normal by Box-Muller
static inline double test_lcg_gauss(unsigned long long *state) {
    double u1 = test_lcg_u01(state);
    double u2 = test_lcg_u01(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

#endif /* MLRISK_TEST_UTIL_H */
