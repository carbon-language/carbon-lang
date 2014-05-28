//===-- lib/adddf3.c - Double-precision addition ------------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements double-precision soft-float addition with the IEEE-754
// default rounding (to nearest, ties to even).
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_add_impl.inc"

ARM_EABI_FNALIAS(dadd, adddf3)

COMPILER_RT_ABI double __adddf3(double a, double b){
    return __addXf3__(a, b);
}
