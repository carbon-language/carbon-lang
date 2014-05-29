//===-- lib/truncdfsf2.c - double -> single conversion ------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SRC_DOUBLE
#define DST_SINGLE
#include "fp_trunc_impl.inc"

ARM_EABI_FNALIAS(d2f, truncdfsf2)

COMPILER_RT_ABI float __truncdfsf2(double a) {
    return __truncXfYf2__(a);
}
