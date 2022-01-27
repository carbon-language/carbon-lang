//===-- Unittests for fabs ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FAbsTest.h"

#include "src/math/fabs.h"

LIST_FABS_TESTS(double, __llvm_libc::fabs)
