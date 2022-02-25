// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divsi3

#include "int_lib.h"
#include <stdio.h>

// Returns: a / b

COMPILER_RT_ABI si_int __divsi3(si_int a, si_int b);

int test__divsi3(si_int a, si_int b, si_int expected)
{
    si_int x = __divsi3(a, b);
    if (x != expected)
        printf("error in __divsi3: %d / %d = %d, expected %d\n",
               a, b, x, expected);
    return x != expected;
}

int main()
{
    if (test__divsi3(0, 1, 0))
        return 1;
    if (test__divsi3(0, -1, 0))
        return 1;

    if (test__divsi3(2, 1, 2))
        return 1;
    if (test__divsi3(2, -1, -2))
        return 1;
    if (test__divsi3(-2, 1, -2))
        return 1;
    if (test__divsi3(-2, -1, 2))
        return 1;

    if (test__divsi3(0x80000000, 1, 0x80000000))
        return 1;
    if (test__divsi3(0x80000000, -1, 0x80000000))
        return 1;
    if (test__divsi3(0x80000000, -2, 0x40000000))
        return 1;
    if (test__divsi3(0x80000000, 2, 0xC0000000))
        return 1;

    return 0;
}
