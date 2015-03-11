/* ===-- fixunsdfdi.c - Implement __fixunsdfdi -----------------------------===
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
typedef du_int fixuint_t;
#include "fp_fixuint_impl.inc"

ARM_EABI_FNALIAS(d2ulz, fixunsdfdi)

COMPILER_RT_ABI du_int
__fixunsdfdi(fp_t a) {
    return __fixuint(a);
}
