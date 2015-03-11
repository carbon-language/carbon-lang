/* ===-- fixunssfsi.c - Implement __fixunssfsi -----------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __fixunssfsi for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#define SINGLE_PRECISION
#include "fp_lib.h"
typedef su_int fixuint_t;
#include "fp_fixuint_impl.inc"

ARM_EABI_FNALIAS(f2uiz, fixunssfsi)

COMPILER_RT_ABI su_int
__fixunssfsi(fp_t a) {
    return __fixuint(a);
}
