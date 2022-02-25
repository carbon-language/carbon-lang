//===-- Unittests for sin -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/TestHelpers.h"
#include "src/math/sin.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcSinTest, Range) {
  static constexpr double _2pi = 6.283185307179586;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = double(FPBits(v));
    // TODO: Expand the range of testing after range reduction is implemented.
    if (isnan(x) || isinf(x) || x > _2pi || x < -_2pi)
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Sin, x, __llvm_libc::sin(x), 1.0);
  }
}
