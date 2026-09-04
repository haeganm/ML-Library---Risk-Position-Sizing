#include <stdio.h>
#include <string.h>

extern int test_rolling(void);
extern int test_vol(void);
extern int test_sizing(void);
extern int test_split(void);
extern int test_linreg(void);
extern int test_fuzz(void);

typedef struct {
    const char *name;
    int (*run)(void);
} test_module;

static const test_module modules[] = {
    {"rolling", test_rolling},
    {"vol", test_vol},
    {"sizing", test_sizing},
    {"split", test_split},
    {"linreg", test_linreg},
    {"fuzz", test_fuzz},
};

// Usage: mlrisk_tests [module ...]   (no arguments runs every module)
int main(int argc, char **argv) {
    int failures = 0;
    int ran = 0;

    for (size_t m = 0; m < sizeof modules / sizeof modules[0]; m++) {
        int selected = argc < 2;
        for (int a = 1; a < argc && !selected; a++) {
            selected = strcmp(argv[a], modules[m].name) == 0;
        }
        if (!selected) {
            continue;
        }
        printf("=== %s ===\n", modules[m].name);
        failures += modules[m].run();
        ran++;
    }

    if (ran == 0) {
        fprintf(stderr, "no such test module\n");
        return 2;
    }
    if (failures == 0) {
        printf("All tests passed\n");
        return 0;
    }
    printf("FAILED: %d test(s)\n", failures);
    return 1;
}
