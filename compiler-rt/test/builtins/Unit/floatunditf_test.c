// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunditf

#include "int_lib.h"
#include <math.h>
#include <complex.h>
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

// Returns: long integer converted to long double

COMPILER_RT_ABI long double __floatunditf(du_int a);

int test__floatunditf(du_int a, uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __floatunditf(a);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret)
        printf("error in __floatunditf(%Lu) = %.20Lf, "
               "expected %.20Lf\n", a, x, fromRep128(expectedHi, expectedLo));
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__floatunditf(0xffffffffffffffffULL, UINT64_C(0x403effffffffffff), UINT64_C(0xfffe000000000000)))
        return 1;
    if (test__floatunditf(0xfffffffffffffffeULL, UINT64_C(0x403effffffffffff), UINT64_C(0xfffc000000000000)))
        return 1;
    if (test__floatunditf(0x8000000000000000ULL, UINT64_C(0x403e000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatunditf(0x7fffffffffffffffULL, UINT64_C(0x403dffffffffffff), UINT64_C(0xfffc000000000000)))
        return 1;
    if (test__floatunditf(0x123456789abcdef1ULL, UINT64_C(0x403b23456789abcd), UINT64_C(0xef10000000000000)))
        return 1;
    if (test__floatunditf(0x2ULL, UINT64_C(0x4000000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatunditf(0x1ULL, UINT64_C(0x3fff000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatunditf(0x0ULL, UINT64_C(0x0), UINT64_C(0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}
