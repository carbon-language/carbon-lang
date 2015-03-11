/* ===-- fixsfdi.c - Implement __fixsfdi -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */

#define SINGLE_PRECISION
#include "fp_lib.h"

ARM_EABI_FNALIAS(f2lz, fixsfdi)

typedef di_int fixint_t;
typedef du_int fixuint_t;
#include "fp_fixint_impl.inc"

COMPILER_RT_ABI di_int
__fixsfdi(fp_t a) {
    return __fixint(a);
}
