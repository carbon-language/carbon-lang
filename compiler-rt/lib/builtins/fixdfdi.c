/* ===-- fixdfdi.c - Implement __fixdfdi -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */

#define DOUBLE_PRECISION
#include "fp_lib.h"
ARM_EABI_FNALIAS(d2lz, fixdfdi)

typedef di_int fixint_t;
typedef du_int fixuint_t;
#include "fp_fixint_impl.inc"

COMPILER_RT_ABI di_int
__fixdfdi(fp_t a) {
    return __fixint(a);
}
