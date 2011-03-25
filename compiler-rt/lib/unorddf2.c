//===-- lib/unorddf2.c - Double-precision comparisons -------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// // This file implements the following soft-float comparison routines:
//
//   __eqdf2   __gedf2   __unorddf2
//   __ledf2   __gtdf2
//   __ltdf2
//   __nedf2
//
// The semantics of the routines grouped in each column are identical, so there
// is a single implementation for each, and wrappers to provide the other names.
//
// The main routines behave as follows:
//
//   __ledf2(a,b) returns -1 if a < b
//                         0 if a == b
//                         1 if a > b
//                         1 if either a or b is NaN
//
//   __gedf2(a,b) returns -1 if a < b
//                         0 if a == b
//                         1 if a > b
//                        -1 if either a or b is NaN
//
//   __unorddf2(a,b) returns 0 if both a and b are numbers
//                           1 if either a or b is NaN
//
// Note that __ledf2( ) and __gedf2( ) are identical except in their handling of
// NaN values.
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

int __unorddf2(fp_t a, fp_t b) {
    const rep_t aAbs = toRep(a) & absMask;
    const rep_t bAbs = toRep(b) & absMask;
    return aAbs > infRep || bAbs > infRep;
}
