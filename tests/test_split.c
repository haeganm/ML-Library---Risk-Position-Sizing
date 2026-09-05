#include "mlrisk/split.h"
#include "test_util.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int test_split_purged_embargoed_known_answer(void) {
    mlr_split splits[8];
    size_t count = 0;

    ASSERT(mlr_walk_forward_splits(60, 20, 10, 10, 3, 5, 1, splits, 8, &count) == MLR_OK,
           "purged split returns MLR_OK");
    ASSERT(count == 4, "n=60,train=20,test=10,step=10 yields 4 splits");

    ASSERT(splits[0].train_start == 0 && splits[0].train_end == 17, "split0 train [0,17)");
    ASSERT(splits[0].test_start == 20 && splits[0].test_end == 30, "split0 test [20,30)");
    ASSERT(splits[0].train_post_start == 35 && splits[0].train_post_end == 60, "split0 post [35,60)");

    ASSERT(splits[2].train_start == 20 && splits[2].train_end == 37, "split2 train [20,37)");
    ASSERT(splits[2].test_start == 40 && splits[2].test_end == 50, "split2 test [40,50)");
    ASSERT(splits[2].train_post_start == 55 && splits[2].train_post_end == 60, "split2 post [55,60)");

    ASSERT(splits[3].test_end == 60, "split3 test_end == n");
    ASSERT(splits[3].train_post_start == splits[3].train_post_end, "embargo past n gives an empty post segment");
    PASS("purged+embargoed known answer");
}

static int test_split_plain_walk_forward(void) {
    // purge = 0, no post segments: contiguous train then test
    mlr_split splits[20];
    size_t count = 0;

    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 0, 0, 0, splits, 20, &count) == MLR_OK, "OK");
    ASSERT(count == 15, "train starts 0,5,...,70 give 15 splits");

    ASSERT(splits[0].train_start == 0 && splits[0].train_end == 20, "split0 train [0,20)");
    ASSERT(splits[0].test_start == 20 && splits[0].test_end == 30, "split0 test [20,30)");
    ASSERT(splits[1].train_start == 5 && splits[1].train_end == 25, "split1 train [5,25)");
    ASSERT(splits[1].test_start == 25 && splits[1].test_end == 35, "split1 test [25,35)");
    for (size_t i = 0; i < count; i++) {
        ASSERT(splits[i].train_post_start == 100 && splits[i].train_post_end == 100,
               "post segments empty without include_post_train");
    }

    ASSERT(mlr_walk_forward_splits(50, 20, 10, 5, 0, 0, 0, splits, 20, &count) == MLR_OK, "OK");
    ASSERT(count == 5, "n=50,train=20,test=10,step=5 yields 5 splits");
    ASSERT(splits[count - 1].test_end == 50, "last split ends exactly at n");
    PASS("plain walk-forward (purge=0)");
}

static int test_split_invariants(void) {
    mlr_split splits[32];
    size_t count = 0;

    ASSERT(mlr_walk_forward_splits(200, 40, 15, 7, 5, 3, 1, splits, 32, &count) == MLR_OK, "OK");
    ASSERT(count == 21, "train starts 0,7,...,140 give 21 splits");

    for (size_t i = 0; i < count; i++) {
        ASSERT(splits[i].train_end == splits[i].test_start - 5, "train_end == test_start - purge");
        ASSERT(splits[i].train_start < splits[i].train_end, "train segment non-empty");
        ASSERT(splits[i].test_start < splits[i].test_end, "test segment non-empty");
        ASSERT(splits[i].test_end <= 200, "test_end within bounds");
        ASSERT(splits[i].train_post_start <= splits[i].train_post_end, "post segment well-formed");
        if (splits[i].train_post_start < splits[i].train_post_end) {
            ASSERT(splits[i].train_post_start >= splits[i].test_end + 3, "post segment respects the embargo");
            ASSERT(splits[i].train_post_end == 200, "post segment runs to n");
        }
    }
    PASS("split invariants");
}

static int test_split_count_query_and_capacity(void) {
    size_t full_count = 0;
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 2, 1, 0, NULL, 0, &full_count) == MLR_OK,
           "count query OK");
    ASSERT(full_count == 15, "count query finds 15 splits");

    mlr_split *splits = (mlr_split *)malloc(full_count * sizeof(mlr_split));
    ASSERT(splits != NULL, "allocation");
    size_t count = 0;
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 2, 1, 0, splits, full_count, &count) == MLR_OK,
           "full-capacity call OK");
    ASSERT(count == full_count, "full-capacity count matches the count query");
    free(splits);

    // Insufficient capacity: exactly `capacity` entries written, EBOUNDS returned
    size_t cap = full_count - 1;
    mlr_split *small = (mlr_split *)malloc((cap + 1) * sizeof(mlr_split));
    ASSERT(small != NULL, "allocation");
    memset(small, 0xAB, (cap + 1) * sizeof(mlr_split));
    mlr_split sentinel = small[cap];

    count = 0;
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 2, 1, 0, small, cap, &count) == MLR_EBOUNDS,
           "insufficient capacity -> EBOUNDS");
    ASSERT(count == full_count, "count_out reports the full required count");
    ASSERT(memcmp(&small[cap], &sentinel, sizeof(mlr_split)) == 0, "entry past capacity untouched");
    ASSERT(small[0].train_start == 0 && small[0].test_start == 20, "entries within capacity written");
    free(small);

    // capacity 0 with a non-NULL buffer writes nothing
    count = 0;
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 2, 1, 0, &sentinel, 0, &count) == MLR_EBOUNDS,
           "capacity 0 -> EBOUNDS");
    ASSERT(count == 15, "capacity 0 still reports the count");
    PASS("count query and capacity handling");
}

static int test_split_count_query_is_closed_form(void) {
    // A count query must not walk every split: n is a size_t and a caller
    // can ask about more samples than exist. These would take years as a
    // loop; they must return at once with the arithmetic answer.
    size_t count = 0;
    ASSERT(mlr_walk_forward_splits(SIZE_MAX, 1, 1, 1, 0, 0, 0, NULL, 0, &count) == MLR_OK,
           "count query at SIZE_MAX OK");
    ASSERT(count == SIZE_MAX - 1, "SIZE_MAX samples, unit windows and step: SIZE_MAX - 1 splits");
    ASSERT(mlr_walk_forward_splits(SIZE_MAX, 10, 5, 3, 0, 0, 0, NULL, 0, &count) == MLR_OK,
           "count query with step 3 OK");
    ASSERT(count == (SIZE_MAX - 15) / 3 + 1, "count is last_train_start / step + 1");

    // And it agrees with the fill pass everywhere the fill pass is affordable
    unsigned long long state = 99;
    for (int trial = 0; trial < 500; trial++) {
        size_t n = 1 + (size_t)(test_lcg_u01(&state) * 60.0);
        size_t train = 1 + (size_t)(test_lcg_u01(&state) * 12.0);
        size_t test = 1 + (size_t)(test_lcg_u01(&state) * 6.0);
        size_t step = 1 + (size_t)(test_lcg_u01(&state) * 5.0);
        size_t purge = (size_t)(test_lcg_u01(&state) * (double)train);
        if (purge >= train) purge = train - 1;
        size_t queried = 0, filled = 0;
        mlr_split buffer[64];
        mlr_status a = mlr_walk_forward_splits(n, train, test, step, purge, 0, 0, NULL, 0, &queried);
        mlr_status b = mlr_walk_forward_splits(n, train, test, step, purge, 0, 0, buffer, 64, &filled);
        ASSERT(a == MLR_OK && b == MLR_OK, "both calls OK");
        ASSERT(queried == filled, "count query equals the fill pass count");
        for (size_t i = 0; i < filled; i++) {
            ASSERT(buffer[i].train_start == i * step, "fill pass writes every split in order");
        }
    }
    PASS("count query is closed form and agrees with the fill pass");
}

static int test_split_invalid_inputs(void) {
    mlr_split splits[4];
    size_t count = 123;

    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 0, 0, 0, splits, 4, NULL) == MLR_EINVAL, "NULL count_out");
    ASSERT(mlr_walk_forward_splits(100, 0, 10, 5, 0, 0, 0, splits, 4, &count) == MLR_EINVAL, "train_len=0");
    ASSERT(count == 0, "count_out is written (0) on EINVAL");
    count = 123;
    ASSERT(mlr_walk_forward_splits(100, 20, 0, 5, 0, 0, 0, splits, 4, &count) == MLR_EINVAL, "test_len=0");
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 0, 0, 0, 0, splits, 4, &count) == MLR_EINVAL, "step=0");
    ASSERT(mlr_walk_forward_splits(100, 20, 10, 5, 20, 0, 0, splits, 4, &count) == MLR_EINVAL, "purge >= train_len");

    // Not enough data is not an error: zero splits
    count = 123;
    ASSERT(mlr_walk_forward_splits(10, 20, 10, 5, 0, 0, 0, splits, 4, &count) == MLR_OK, "train+test > n OK");
    ASSERT(count == 0, "train+test > n yields zero splits");
    count = 123;
    ASSERT(mlr_walk_forward_splits(0, 20, 10, 5, 0, 0, 0, splits, 4, &count) == MLR_OK, "n=0 OK");
    ASSERT(count == 0, "n=0 yields zero splits");
    PASS("split invalid inputs");
}

static int test_split_no_wraparound(void) {
    // Arguments near SIZE_MAX must not wrap into bogus in-bounds splits
    mlr_split splits[4];
    size_t count = 123;

    ASSERT(mlr_walk_forward_splits(100, SIZE_MAX - 4, 5, 1, 0, 0, 0, splits, 4, &count) == MLR_OK,
           "huge train_len OK");
    ASSERT(count == 0, "huge train_len yields zero splits, not wrapped ones");

    ASSERT(mlr_walk_forward_splits(100, 5, SIZE_MAX - 4, 1, 0, 0, 0, splits, 4, &count) == MLR_OK,
           "huge test_len OK");
    ASSERT(count == 0, "huge test_len yields zero splits");

    // A huge embargo means no post-test data, never a post segment inside the test window
    ASSERT(mlr_walk_forward_splits(60, 20, 10, 10, 3, SIZE_MAX, 1, splits, 4, &count) == MLR_OK,
           "huge embargo OK");
    ASSERT(count == 4, "huge embargo does not change the split count");
    for (size_t i = 0; i < count; i++) {
        ASSERT(splits[i].train_post_start == 60 && splits[i].train_post_end == 60,
               "huge embargo gives empty post segments");
    }
    ASSERT(mlr_walk_forward_splits(60, 20, 10, 10, 3, SIZE_MAX - 25, 1, splits, 4, &count) == MLR_OK, "OK");
    ASSERT(splits[0].train_post_start == 60, "wrapping embargo gives an empty post segment");

    // A huge step yields exactly one split
    ASSERT(mlr_walk_forward_splits(100, 20, 10, SIZE_MAX, 0, 0, 0, splits, 4, &count) == MLR_OK, "huge step OK");
    ASSERT(count == 1, "huge step yields one split");
    PASS("no size_t wraparound");
}

int test_split(void) {
    int failures = 0;
    failures += test_split_purged_embargoed_known_answer();
    failures += test_split_plain_walk_forward();
    failures += test_split_invariants();
    failures += test_split_count_query_and_capacity();
    failures += test_split_count_query_is_closed_form();
    failures += test_split_invalid_inputs();
    failures += test_split_no_wraparound();
    return failures;
}
