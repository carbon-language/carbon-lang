/* ===-- fixsfsi.c - Implement __fixsfsi -----------------------------------===
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
typedef si_int fixint_t;
typedef su_int fixuint_t;
#include "fp_fixint_impl.inc"

ARM_EABI_FNALIAS(f2iz, fixsfsi)

COMPILER_RT_ABI si_int
__fixsfsi(fp_t a) {
    return __fixint(a);
}
