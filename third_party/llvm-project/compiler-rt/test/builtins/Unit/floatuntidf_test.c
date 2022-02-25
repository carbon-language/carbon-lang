// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatuntidf
// REQUIRES: int128

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: convert a to a double, rounding toward even.

// Assumption: double is a IEEE 64 bit floating point type 
//             tu_int is a 64 bit integral type

// seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI double __floatuntidf(tu_int a);

int test__floatuntidf(tu_int a, double expected)
{
    double x = __floatuntidf(a);
    if (x != expected)
    {
        utwords at;
        at.all = a;
        printf("error in __floatuntidf(0x%.16llX%.16llX) = %a, expected %a\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(ti_int)*CHAR_BIT == 128] = {0};
char assumption_3[sizeof(double)*CHAR_BIT == 64] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__floatuntidf(0, 0.0))
        return 1;

    if (test__floatuntidf(1, 1.0))
        return 1;
    if (test__floatuntidf(2, 2.0))
        return 1;
    if (test__floatuntidf(20, 20.0))
        return 1;

    if (test__floatuntidf(0x7FFFFF8000000000LL, 0x1.FFFFFEp+62))
        return 1;
    if (test__floatuntidf(0x7FFFFFFFFFFFF800LL, 0x1.FFFFFFFFFFFFEp+62))
        return 1;
    if (test__floatuntidf(0x7FFFFF0000000000LL, 0x1.FFFFFCp+62))
        return 1;
    if (test__floatuntidf(0x7FFFFFFFFFFFF000LL, 0x1.FFFFFFFFFFFFCp+62))
        return 1;

    if (test__floatuntidf(make_ti(0x8000008000000000LL, 0), 0x1.000001p+127))
        return 1;
    if (test__floatuntidf(make_ti(0x8000000000000800LL, 0), 0x1.0000000000001p+127))
        return 1;
    if (test__floatuntidf(make_ti(0x8000010000000000LL, 0), 0x1.000002p+127))
        return 1;
    if (test__floatuntidf(make_ti(0x8000000000001000LL, 0), 0x1.0000000000002p+127))
        return 1;

    if (test__floatuntidf(make_ti(0x8000000000000000LL, 0), 0x1.000000p+127))
        return 1;
    if (test__floatuntidf(make_ti(0x8000000000000001LL, 0), 0x1.0000000000000002p+127))
        return 1;

    if (test__floatuntidf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50))
        return 1;

    if (test__floatuntidf(0x0007FB72EA000000LL, 0x1.FEDCBA8p+50))
        return 1;
    if (test__floatuntidf(0x0007FB72EB000000LL, 0x1.FEDCBACp+50))
        return 1;
    if (test__floatuntidf(0x0007FB72EBFFFFFFLL, 0x1.FEDCBAFFFFFFCp+50))
        return 1;
    if (test__floatuntidf(0x0007FB72EC000000LL, 0x1.FEDCBBp+50))
        return 1;
    if (test__floatuntidf(0x0007FB72E8000001LL, 0x1.FEDCBA0000004p+50))
        return 1;

    if (test__floatuntidf(0x0007FB72E6000000LL, 0x1.FEDCB98p+50))
        return 1;
    if (test__floatuntidf(0x0007FB72E7000000LL, 0x1.FEDCB9Cp+50))
        return 1;
    if (test__floatuntidf(0x0007FB72E7FFFFFFLL, 0x1.FEDCB9FFFFFFCp+50))
        return 1;
    if (test__floatuntidf(0x0007FB72E4000001LL, 0x1.FEDCB90000004p+50))
        return 1;
    if (test__floatuntidf(0x0007FB72E4000000LL, 0x1.FEDCB9p+50))
        return 1;

    if (test__floatuntidf(0x023479FD0E092DC0LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DA1LL, 0x1.1A3CFE870496Dp+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DB0LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DB8LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DB6LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DBFLL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DC1LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DC7LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DC8LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DCFLL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DD0LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DD1LL, 0x1.1A3CFE870496Fp+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DD8LL, 0x1.1A3CFE870496Fp+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DDFLL, 0x1.1A3CFE870496Fp+57))
        return 1;
    if (test__floatuntidf(0x023479FD0E092DE0LL, 0x1.1A3CFE870496Fp+57))
        return 1;

    if (test__floatuntidf(make_ti(0x023479FD0E092DC0LL, 0), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DA1LL, 1), 0x1.1A3CFE870496Dp+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DB0LL, 2), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DB8LL, 3), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DB6LL, 4), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DBFLL, 5), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DC1LL, 6), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DC7LL, 7), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DC8LL, 8), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DCFLL, 9), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DD0LL, 0), 0x1.1A3CFE870496Ep+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DD1LL, 11), 0x1.1A3CFE870496Fp+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DD8LL, 12), 0x1.1A3CFE870496Fp+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DDFLL, 13), 0x1.1A3CFE870496Fp+121))
        return 1;
    if (test__floatuntidf(make_ti(0x023479FD0E092DE0LL, 14), 0x1.1A3CFE870496Fp+121))
        return 1;
#else
    printf("skipped\n");
#endif
   return 0;
}
