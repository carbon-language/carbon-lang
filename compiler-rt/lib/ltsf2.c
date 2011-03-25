//===-- lib/comparesf2.c - Single-precision comparisons -----------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

enum LE_RESULT __lesf2(fp_t a, fp_t b);
enum LE_RESULT __ltsf2(fp_t a, fp_t b) {
    return __lesf2(a, b);
}
