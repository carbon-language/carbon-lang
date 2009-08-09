/* ===-- ffsti2.c - Implement __ffsti2 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __ffsti2 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if __x86_64

#include "int_lib.h"

/* Returns: the index of the least significant 1-bit in a, or
 * the value zero if a is zero. The least significant bit is index one.
 */

si_int
__ffsti2(ti_int a)
{
    twords x;
    x.all = a;
    if (x.low == 0)
    {
        if (x.high == 0)
            return 0;
        return __builtin_ctzll(x.high) + (1 + sizeof(di_int) * CHAR_BIT);
    }
    return __builtin_ctzll(x.low) + 1;
}

#endif /* __x86_64 */
