#ifndef MLRISK_TYPES_H
#define MLRISK_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include <math.h>

/**
 * @file types.h
 * @brief Core types, error codes, and utility functions for mlrisk library
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes for mlrisk functions
 */
typedef enum {
    MLR_OK = 0,        /**< Success */
    MLR_EINVAL = 1,    /**< Invalid argument */
    MLR_ENOMEM = 2,    /**< Memory allocation failure */
    MLR_EBOUNDS = 3,   /**< Index out of bounds */
    MLR_EDOMAIN = 4    /**< Domain error (e.g., negative value where positive required) */
} mlr_status;

/**
 * @brief Not-a-Number constant for mlrisk
 */
#define MLR_NAN NAN

/**
 * @brief Output arrays must not alias input arrays
 *
 * Every function that writes an output array reads its inputs while it
 * writes, so passing the same buffer as input and output corrupts the
 * result. The `restrict` qualifier on output parameters states that
 * contract to the compiler; it is spelled through this macro so the headers
 * also compile as C++.
 */
#ifdef __cplusplus
#define MLR_RESTRICT
#else
#define MLR_RESTRICT restrict
#endif

/**
 * @brief Check if a value is NaN
 * @param x Value to check
 * @return true if x is NaN, false otherwise
 */
static inline bool mlr_isnan(double x) {
    return isnan(x);
}

/**
 * @brief Check if a value is finite (not NaN or Inf)
 * @param x Value to check
 * @return true if x is finite, false otherwise
 */
static inline bool mlr_isfinite(double x) {
    return isfinite(x);
}

#ifdef __cplusplus
}
#endif

#endif /* MLRISK_TYPES_H */
