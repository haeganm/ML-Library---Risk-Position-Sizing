#include "mlrisk/split.h"

mlr_status mlr_walk_forward_splits(
    size_t n,
    size_t train_len,
    size_t test_len,
    size_t step,
    size_t purge,
    size_t embargo,
    int include_post_train,
    mlr_split *splits_out,
    size_t capacity,
    size_t *count_out
) {
    if (count_out == NULL) {
        return MLR_EINVAL;
    }
    if (train_len == 0 || test_len == 0 || step == 0) {
        return MLR_EINVAL;
    }
    if (purge >= train_len) {
        return MLR_EINVAL;
    }

    // Not enough data for a single split. Checked this way so that the loop
    // arithmetic below cannot wrap around.
    if (train_len > n || test_len > n - train_len) {
        *count_out = 0;
        return MLR_OK;
    }
    size_t last_train_start = n - train_len - test_len;

    size_t count = 0;
    for (size_t train_start = 0; train_start <= last_train_start; train_start += step) {
        if (splits_out != NULL && count < capacity) {
            mlr_split s;
            s.train_start = train_start;
            s.test_start = train_start + train_len;
            s.test_end = s.test_start + test_len;
            s.train_end = s.test_start - purge;

            // test_end <= n, so n - test_end cannot wrap
            if (include_post_train && embargo < n - s.test_end) {
                s.train_post_start = s.test_end + embargo;
            } else {
                s.train_post_start = n;
            }
            s.train_post_end = n;

            splits_out[count] = s;
        }
        count++;
        if (step > last_train_start - train_start) {
            break;
        }
    }

    *count_out = count;
    if (splits_out != NULL && count > capacity) {
        return MLR_EBOUNDS;
    }
    return MLR_OK;
}
