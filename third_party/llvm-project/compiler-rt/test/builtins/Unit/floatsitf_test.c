// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatsitf

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

COMPILER_RT_ABI long double __floatsitf(si_int a);

int test__floatsitf(si_int a, uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __floatsitf(a);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret)
    {
        printf("error in test__floatsitf(%d) = %.20Lf, "
               "expected %.20Lf\n", a, x, fromRep128(expectedHi, expectedLo));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__floatsitf(0x80000000, UINT64_C(0xc01e000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatsitf(0x7fffffff, UINT64_C(0x401dfffffffc0000), UINT64_C(0x0)))
        return 1;
    if (test__floatsitf(0, UINT64_C(0x0), UINT64_C(0x0)))
        return 1;
    if (test__floatsitf(0xffffffff, UINT64_C(0xbfff000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatsitf(0x12345678, UINT64_C(0x401b234567800000), UINT64_C(0x0)))
        return 1;
    if (test__floatsitf(-0x12345678, UINT64_C(0xc01b234567800000), UINT64_C(0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}
