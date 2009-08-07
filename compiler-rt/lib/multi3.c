/* ===-- multi3.c - Implement __multi3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===

 * This file implements __multi3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if __x86_64

#include "int_lib.h"

/* Returns: a * b */

static
ti_int
__mulddi3(du_int a, du_int b)
{
    twords r;
    const int bits_in_dword_2 = (int)(sizeof(di_int) * CHAR_BIT) / 2;
    const du_int lower_mask = (du_int)~0 >> bits_in_dword_2;
    r.low = (a & lower_mask) * (b & lower_mask);
    du_int t = r.low >> bits_in_dword_2;
    r.low &= lower_mask;
    t += (a >> bits_in_dword_2) * (b & lower_mask);
    r.low += (t & lower_mask) << bits_in_dword_2;
    r.high = t >> bits_in_dword_2;
    t = r.low >> bits_in_dword_2;
    r.low &= lower_mask;
    t += (b >> bits_in_dword_2) * (a & lower_mask);
    r.low += (t & lower_mask) << bits_in_dword_2;
    r.high += t >> bits_in_dword_2;
    r.high += (a >> bits_in_dword_2) * (b >> bits_in_dword_2);
    return r.all;
}

/* Returns: a * b */

ti_int
__multi3(ti_int a, ti_int b)
{
    twords x;
    x.all = a;
    twords y;
    y.all = b;
    twords r;
    r.all = __mulddi3(x.low, y.low);
    r.high += x.high * y.low + x.low * y.high;
    return r.all;
}

#endif
