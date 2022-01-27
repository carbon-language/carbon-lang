// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ffssi2

#include "int_lib.h"
#include <stdio.h>

// Returns: the index of the least significant 1-bit in a, or
// the value zero if a is zero. The least significant bit is index one.

COMPILER_RT_ABI int __ffssi2(si_int a);

int test__ffssi2(si_int a, int expected)
{
    int x = __ffssi2(a);
    if (x != expected)
        printf("error in __ffssi2(0x%X) = %d, expected %d\n", a, x, expected);
    return x != expected;
}

int main()
{
    if (test__ffssi2(0x00000000, 0))
        return 1;
    if (test__ffssi2(0x00000001, 1))
        return 1;
    if (test__ffssi2(0x00000002, 2))
        return 1;
    if (test__ffssi2(0x00000003, 1))
        return 1;
    if (test__ffssi2(0x00000004, 3))
        return 1;
    if (test__ffssi2(0x00000005, 1))
        return 1;
    if (test__ffssi2(0x0000000A, 2))
        return 1;
    if (test__ffssi2(0x10000000, 29))
        return 1;
    if (test__ffssi2(0x20000000, 30))
        return 1;
    if (test__ffssi2(0x60000000, 30))
        return 1;
    if (test__ffssi2(0x80000000u, 32))
        return 1;

   return 0;
}
