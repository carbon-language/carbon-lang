//===-- Unittests for fmaxf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMaxTest.h"

#include "src/math/fmaxf.h"

LIST_FMAX_TESTS(float, __llvm_libc::fmaxf)
