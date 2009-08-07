/* ===-- ashlti3.c - Implement __ashlti3 -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __ashlti3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if __x86_64

#include "int_lib.h"

/* Returns: a << b */

/* Precondition:  0 <= b < bits_in_tword */

ti_int
__ashlti3(ti_int a, si_int b)
{
    const int bits_in_dword = (int)(sizeof(di_int) * CHAR_BIT);
    twords input;
    twords result;
    input.all = a;
    if (b & bits_in_dword)  /* bits_in_dword <= b < bits_in_tword */
    {
        result.low = 0;
        result.high = input.low << (b - bits_in_dword);
    }
    else  /* 0 <= b < bits_in_dword */
    {
        if (b == 0)
            return a;
        result.low  = input.low << b;
        result.high = (input.high << b) | (input.low >> (bits_in_dword - b));
    }
    return result.all;
}

#endif /* __x86_64 */
