//===-- lib/nedf2.c - Double-precision comparisons ----------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

enum LE_RESULT __ledf2(fp_t a, fp_t b);
enum LE_RESULT __nedf2(fp_t a, fp_t b) {
    return __ledf2(a, b);
}

