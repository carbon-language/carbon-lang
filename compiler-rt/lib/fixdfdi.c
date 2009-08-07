/* ===-- fixdfdi.c - Implement __fixdfdi -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __fixdfdi for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

/* Returns: convert a to a signed long long, rounding toward zero. */

/* Assumption: double is a IEEE 64 bit floating point type 
 *            su_int is a 32 bit integral type
 *            value in double is representable in di_int (no range checking performed)
 */

/* seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm */

di_int
__fixdfdi(double a)
{
    double_bits fb;
    fb.f = a;
    int e = ((fb.u.high & 0x7FF00000) >> 20) - 1023;
    if (e < 0)
        return 0;
    di_int s = (si_int)(fb.u.high & 0x80000000) >> 31;
    dwords r;
    r.high = (fb.u.high & 0x000FFFFF) | 0x00100000;
    r.low = fb.u.low;
    if (e > 52)
        r.all <<= (e - 52);
    else
        r.all >>= (52 - e);
    return (r.all ^ s) - s;
} 
