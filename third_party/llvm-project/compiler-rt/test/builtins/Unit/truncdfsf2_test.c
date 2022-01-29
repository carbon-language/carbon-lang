// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncdfsf2

#include <stdio.h>

#include "fp_test.h"

float __truncdfsf2(double a);

int test__truncdfsf2(double a)
{
    float actual = __truncdfsf2(a);
    float expected = a;

    if (actual != expected) {
        printf("error in test__truncdfsf2(%lf) = %f, "
               "expected %f\n", a, actual, expected);
        return 1;
    }
    return 0;
}

int main()
{
    if (test__truncdfsf2(340282366920938463463374607431768211456.0))
        return 1;
    return 0;
}
