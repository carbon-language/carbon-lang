//===-- lib/truncsfhf2.c - single -> half conversion --------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SRC_SINGLE
#define DST_HALF
#include "fp_trunc_impl.inc"

COMPILER_RT_ABI uint16_t __truncsfhf2(float a) {
    return __truncXfYf2__(a);
}

COMPILER_RT_ABI uint16_t __gnu_f2h_ieee(float a) {
    return __truncXfYf2__(a);
}
