//===-- lib/muldf3.c - Double-precision multiplication ------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements double-precision soft-float multiplication
// with the IEEE-754 default rounding (to nearest, ties to even).
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_mul_impl.inc"

ARM_EABI_FNALIAS(dmul, muldf3)

COMPILER_RT_ABI fp_t __muldf3(fp_t a, fp_t b) {
    return __mulXf3__(a, b);
}
