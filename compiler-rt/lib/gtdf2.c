//===-- lib/gtdf2.c - Double-precision comparisons ----------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

enum GE_RESULT __gedf2(fp_t a, fp_t b);
enum GE_RESULT __gtdf2(fp_t a, fp_t b) {
    return __gedf2(a, b);
}

