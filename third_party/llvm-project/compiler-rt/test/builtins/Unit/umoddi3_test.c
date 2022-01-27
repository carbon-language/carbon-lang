// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_umoddi3

#include "int_lib.h"
#include <stdio.h>

// Returns: a % b

COMPILER_RT_ABI du_int __umoddi3(du_int a, du_int b);

int test__umoddi3(du_int a, du_int b, du_int expected_r)
{
    du_int r = __umoddi3(a, b);
    if (r != expected_r)
        printf("error in __umoddi3: %lld %% %lld = %lld, expected %lld\n",
               a, b, r, expected_r);
    return r != expected_r;
}

int main()
{
    if (test__umoddi3(0, 1, 0))
        return 1;
    if (test__umoddi3(2, 1, 0))
        return 1;
    if (test__umoddi3(0x8000000000000000uLL, 1, 0x0uLL))
        return 1;
    if (test__umoddi3(0x8000000000000000uLL, 2, 0x0uLL))
        return 1;
    if (test__umoddi3(0xFFFFFFFFFFFFFFFFuLL, 2, 0x1uLL))
        return 1;

    return 0;
}
