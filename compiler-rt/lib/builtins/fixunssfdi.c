/* ===-- fixunssfdi.c - Implement __fixunssfdi -----------------------------===
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
typedef du_int fixuint_t;
#include "fp_fixuint_impl.inc"

ARM_EABI_FNALIAS(f2ulz, fixunssfdi)

COMPILER_RT_ABI du_int
__fixunssfdi(fp_t a) {
    return __fixuint(a);
}
