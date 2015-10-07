//===-- lib/truncdfhf2.c - double -> half conversion --------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SRC_DOUBLE
#define DST_HALF
#include "fp_trunc_impl.inc"

ARM_EABI_FNALIAS(d2h, truncdfhf2)

COMPILER_RT_ABI uint16_t __truncdfhf2(double a) {
    return __truncXfYf2__(a);
}
