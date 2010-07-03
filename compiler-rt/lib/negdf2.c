//===-- lib/negdf2.c - double-precision negation ------------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements double-precision soft-float negation.
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

fp_t __negdf2(fp_t a) {
    return fromRep(toRep(a) ^ signBit);
}
