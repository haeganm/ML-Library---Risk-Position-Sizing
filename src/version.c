#include "mlrisk/version.h"

const char *mlr_version(void) {
    return MLRISK_VERSION;
}

int mlr_version_number(void) {
    return MLRISK_VERSION_MAJOR * 10000 + MLRISK_VERSION_MINOR * 100 + MLRISK_VERSION_PATCH;
}
