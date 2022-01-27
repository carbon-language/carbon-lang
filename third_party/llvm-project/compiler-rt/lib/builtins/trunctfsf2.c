//===-- lib/trunctfsf2.c - quad -> single conversion --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT)
#define SRC_QUAD
#define DST_SINGLE
#include "fp_trunc_impl.inc"

COMPILER_RT_ABI float __trunctfsf2(long double a) { return __truncXfYf2__(a); }

#endif
