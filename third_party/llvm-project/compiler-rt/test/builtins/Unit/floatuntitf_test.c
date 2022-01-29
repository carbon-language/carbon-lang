// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatuntitf

#define QUAD_PRECISION
#include "fp_lib.h"
#include "int_lib.h"
#include <float.h>
#include <stdio.h>

#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT)

/* Returns: convert a tu_int to a fp_t, rounding toward even. */

/* Assumption: fp_t is a IEEE 128 bit floating point type
 *             tu_int is a 128 bit integral type
 */

/* seee eeee eeee eeee mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm |
 * mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm
 */

COMPILER_RT_ABI fp_t __floatuntitf(tu_int a);

int test__floatuntitf(tu_int a, fp_t expected) {
    fp_t x = __floatuntitf(a);
    if (x != expected) {
        utwords at;
        at.all = a;
        printf("error in __floatuntitf(0x%.16llX%.16llX) = %LA, expected %LA\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(tu_int) == 2*sizeof(du_int)] = {0};
char assumption_2[sizeof(tu_int)*CHAR_BIT == 128] = {0};
char assumption_3[sizeof(fp_t)*CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT)
    if (test__floatuntitf(0, 0.0))
        return 1;

    if (test__floatuntitf(1, 1.0))
        return 1;
    if (test__floatuntitf(2, 2.0))
        return 1;
    if (test__floatuntitf(20, 20.0))
        return 1;

    if (test__floatuntitf(0x7FFFFF8000000000ULL, 0x1.FFFFFEp+62))
        return 1;
    if (test__floatuntitf(0x7FFFFFFFFFFFF800ULL, 0x1.FFFFFFFFFFFFEp+62))
        return 1;
    if (test__floatuntitf(0x7FFFFF0000000000ULL, 0x1.FFFFFCp+62))
        return 1;
    if (test__floatuntitf(0x7FFFFFFFFFFFF000ULL, 0x1.FFFFFFFFFFFFCp+62))
        return 1;
    if (test__floatuntitf(0x7FFFFFFFFFFFFFFFULL, 0xF.FFFFFFFFFFFFFFEp+59L))
        return 1;
    if (test__floatuntitf(0xFFFFFFFFFFFFFFFEULL, 0xF.FFFFFFFFFFFFFFEp+60L))
        return 1;
    if (test__floatuntitf(0xFFFFFFFFFFFFFFFFULL, 0xF.FFFFFFFFFFFFFFFp+60L))
        return 1;

    if (test__floatuntitf(0x8000008000000000ULL, 0x8.000008p+60))
        return 1;
    if (test__floatuntitf(0x8000000000000800ULL, 0x8.0000000000008p+60))
        return 1;
    if (test__floatuntitf(0x8000010000000000ULL, 0x8.00001p+60))
        return 1;
    if (test__floatuntitf(0x8000000000001000ULL, 0x8.000000000001p+60))
        return 1;

    if (test__floatuntitf(0x8000000000000000ULL, 0x8p+60))
        return 1;
    if (test__floatuntitf(0x8000000000000001ULL, 0x8.000000000000001p+60L))
        return 1;

    if (test__floatuntitf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50))
        return 1;

    if (test__floatuntitf(0x0007FB72EA000000LL, 0x1.FEDCBA8p+50))
        return 1;
    if (test__floatuntitf(0x0007FB72EB000000LL, 0x1.FEDCBACp+50))
        return 1;
    if (test__floatuntitf(0x0007FB72EBFFFFFFLL, 0x1.FEDCBAFFFFFFCp+50))
        return 1;
    if (test__floatuntitf(0x0007FB72EC000000LL, 0x1.FEDCBBp+50))
        return 1;
    if (test__floatuntitf(0x0007FB72E8000001LL, 0x1.FEDCBA0000004p+50))
        return 1;

    if (test__floatuntitf(0x0007FB72E6000000LL, 0x1.FEDCB98p+50))
        return 1;
    if (test__floatuntitf(0x0007FB72E7000000LL, 0x1.FEDCB9Cp+50))
        return 1;
    if (test__floatuntitf(0x0007FB72E7FFFFFFLL, 0x1.FEDCB9FFFFFFCp+50))
        return 1;
    if (test__floatuntitf(0x0007FB72E4000001LL, 0x1.FEDCB90000004p+50))
        return 1;
    if (test__floatuntitf(0x0007FB72E4000000LL, 0x1.FEDCB9p+50))
        return 1;

    if (test__floatuntitf(0x023479FD0E092DC0LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DA1LL, 0x1.1A3CFE870496D08p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DB0LL, 0x1.1A3CFE870496D8p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DB8LL, 0x1.1A3CFE870496DCp+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DB6LL, 0x1.1A3CFE870496DBp+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DBFLL, 0x1.1A3CFE870496DF8p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DC1LL, 0x1.1A3CFE870496E08p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DC7LL, 0x1.1A3CFE870496E38p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DC8LL, 0x1.1A3CFE870496E4p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DCFLL, 0x1.1A3CFE870496E78p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DD0LL, 0x1.1A3CFE870496E8p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DD1LL, 0x1.1A3CFE870496E88p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DD8LL, 0x1.1A3CFE870496ECp+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DDFLL, 0x1.1A3CFE870496EF8p+57L))
        return 1;
    if (test__floatuntitf(0x023479FD0E092DE0LL, 0x1.1A3CFE870496Fp+57))
        return 1;

    if (test__floatuntitf(make_ti(0x023479FD0E092DC0LL, 0), 0x1.1A3CFE870496Ep+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DA1LL, 1), 0x1.1A3CFE870496D08p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DB0LL, 2), 0x1.1A3CFE870496D8p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DB8LL, 3), 0x1.1A3CFE870496DCp+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DB6LL, 4), 0x1.1A3CFE870496DBp+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DBFLL, 5), 0x1.1A3CFE870496DF8p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DC1LL, 6), 0x1.1A3CFE870496E08p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DC7LL, 7), 0x1.1A3CFE870496E38p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DC8LL, 8), 0x1.1A3CFE870496E4p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DCFLL, 9), 0x1.1A3CFE870496E78p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DD0LL, 0), 0x1.1A3CFE870496E8p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DD1LL, 11), 0x1.1A3CFE870496E88p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DD8LL, 12), 0x1.1A3CFE870496ECp+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DDFLL, 13), 0x1.1A3CFE870496EF8p+121L))
        return 1;
    if (test__floatuntitf(make_ti(0x023479FD0E092DE0LL, 14), 0x1.1A3CFE870496Fp+121L))
        return 1;

    if (test__floatuntitf(make_ti(0, 0xFFFFFFFFFFFFFFFFLL), 0x1.FFFFFFFFFFFFFFFEp+63L))
        return 1;

    if (test__floatuntitf(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x0000000000000000LL),
                          0x1.FFFFFFFFFFFFFFFEp+127L))
        return 1;
    if (test__floatuntitf(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                          0x1.0000000000000000p+128L))
        return 1;

    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC2801LL),
                        0x1.23456789ABCDEF0123456789ABC3p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC3000LL),
                        0x1.23456789ABCDEF0123456789ABC3p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC37FFLL),
                        0x1.23456789ABCDEF0123456789ABC3p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC3800LL),
                        0x1.23456789ABCDEF0123456789ABC4p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC4000LL),
                        0x1.23456789ABCDEF0123456789ABC4p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC47FFLL),
                        0x1.23456789ABCDEF0123456789ABC4p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC4800LL),
                        0x1.23456789ABCDEF0123456789ABC4p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC4801LL),
                        0x1.23456789ABCDEF0123456789ABC5p+124L))
        return 1;
    if (test__floatuntitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC57FFLL),
                        0x1.23456789ABCDEF0123456789ABC5p+124L))
        return 1;
#else
    printf("skipped\n");
#endif
   return 0;
}
