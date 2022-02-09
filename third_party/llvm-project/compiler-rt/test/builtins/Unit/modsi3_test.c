// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_modsi3

#include "int_lib.h"
#include <stdio.h>

/* Returns: a % b */

COMPILER_RT_ABI si_int __modsi3(si_int a, si_int b);

int test__modsi3(si_int a, si_int b, si_int expected) {
    si_int x = __modsi3(a, b);
    if (x != expected)
        fprintf(stderr, "error in __modsi3: %d %% %d = %d, expected %d\n",
               a, b, x, expected);
    return x != expected;
}

int main() {
    if (test__modsi3(0, 1, 0))
        return 1;
    if (test__modsi3(0, -1, 0))
        return 1;

    if (test__modsi3(5, 3, 2))
        return 1;
    if (test__modsi3(5, -3, 2))
        return 1;
    if (test__modsi3(-5, 3, -2))
        return 1;
    if (test__modsi3(-5, -3, -2))
        return 1;

    if (test__modsi3(0x80000000, 1, 0x0))
        return 1;
    if (test__modsi3(0x80000000, 2, 0x0))
        return 1;
    if (test__modsi3(0x80000000, -2, 0x0))
        return 1;
    if (test__modsi3(0x80000000, 3, -2))
        return 1;
    if (test__modsi3(0x80000000, -3, -2))
        return 1;

    return 0;
}
