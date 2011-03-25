//===-- lib/gtsf2.c - Single-precision comparisons ----------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

enum GE_RESULT __gesf2(fp_t a, fp_t b);
enum GE_RESULT __gtsf2(fp_t a, fp_t b) {
    return __gesf2(a, b);
}
