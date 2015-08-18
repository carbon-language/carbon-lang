//===-- lib/arm/aeabi_drsub.c - Double-precision subtraction --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "../fp_lib.h"

COMPILER_RT_ABI fp_t
__aeabi_dsub(fp_t, fp_t);

COMPILER_RT_ABI fp_t
__aeabi_drsub(fp_t a, fp_t b) {
    return __aeabi_dsub(b, a);
}
