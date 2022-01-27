// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_moddi3

#include "int_lib.h"
#include <stdio.h>

// Returns: a % b

COMPILER_RT_ABI di_int __moddi3(di_int a, di_int b);

int test__moddi3(di_int a, di_int b, di_int expected)
{
    di_int x = __moddi3(a, b);
    if (x != expected)
        printf("error in __moddi3: %lld %% %lld = %lld, expected %lld\n",
               a, b, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__moddi3(0, 1, 0))
        return 1;
    if (test__moddi3(0, -1, 0))
        return 1;

    if (test__moddi3(5, 3, 2))
        return 1;
    if (test__moddi3(5, -3, 2))
        return 1;
    if (test__moddi3(-5, 3, -2))
        return 1;
    if (test__moddi3(-5, -3, -2))
        return 1;

    if (test__moddi3(0x8000000000000000LL, 1, 0x0LL))
        return 1;
    if (test__moddi3(0x8000000000000000LL, -1, 0x0LL))
        return 1;
    if (test__moddi3(0x8000000000000000LL, 2, 0x0LL))
        return 1;
    if (test__moddi3(0x8000000000000000LL, -2, 0x0LL))
        return 1;
    if (test__moddi3(0x8000000000000000LL, 3, -2))
        return 1;
    if (test__moddi3(0x8000000000000000LL, -3, -2))
        return 1;

    return 0;
}
