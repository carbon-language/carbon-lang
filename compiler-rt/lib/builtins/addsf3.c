//===-- lib/addsf3.c - Single-precision addition ------------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements single-precision soft-float addition with the IEEE-754
// default rounding (to nearest, ties to even).
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_add_impl.inc"

ARM_EABI_FNALIAS(fadd, addsf3)

COMPILER_RT_ABI float __addsf3(float a, float b) {
    return __addXf3__(a, b);
}
