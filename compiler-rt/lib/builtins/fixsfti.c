/* ===-- fixsfti.c - Implement __fixsfti -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __fixsfti for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

#ifdef CRT_HAS_128BIT

/* Returns: convert a to a signed long long, rounding toward zero. */

/* Assumption: float is a IEEE 32 bit floating point type 
 *             su_int is a 32 bit integral type
 *             value in float is representable in ti_int (no range checking performed)
 */

/* seee eeee emmm mmmm mmmm mmmm mmmm mmmm */

COMPILER_RT_ABI ti_int
__fixsfti(float a)
{
    float_bits fb;
    fb.f = a;
    int e = ((fb.u & 0x7F800000) >> 23) - 127;
    if (e < 0)
        return 0;
    ti_int s = (si_int)(fb.u & 0x80000000) >> 31;
    ti_int r = (fb.u & 0x007FFFFF) | 0x00800000;
    if (e > 23)
        r <<= (e - 23);
    else
        r >>= (23 - e);
    return (r ^ s) - s;
}

#endif /* CRT_HAS_128BIT */
