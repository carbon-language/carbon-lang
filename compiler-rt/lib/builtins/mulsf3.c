//===-- lib/mulsf3.c - Single-precision multiplication ------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements single-precision soft-float multiplication
// with the IEEE-754 default rounding (to nearest, ties to even).
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_mul_impl.inc"

ARM_EABI_FNALIAS(fmul, mulsf3)

COMPILER_RT_ABI fp_t __mulsf3(fp_t a, fp_t b) {
    return __mulXf3__(a, b);
}
