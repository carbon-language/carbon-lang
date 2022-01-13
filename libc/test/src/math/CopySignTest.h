//===-- Utility class to test copysign[f|l] ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T> class CopySignTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CopySignFunc)(T, T);

  void testSpecialNumbers(CopySignFunc func) {
    EXPECT_FP_EQ(aNaN, func(aNaN, -1.0));
    EXPECT_FP_EQ(aNaN, func(aNaN, 1.0));

    EXPECT_FP_EQ(neg_inf, func(inf, -1.0));
    EXPECT_FP_EQ(inf, func(neg_inf, 1.0));

    EXPECT_FP_EQ(neg_zero, func(zero, -1.0));
    EXPECT_FP_EQ(zero, func(neg_zero, 1.0));
  }

  void testRange(CopySignFunc func) {
    constexpr UIntType COUNT = 10000000;
    constexpr UIntType STEP = UIntType(-1) / COUNT;
    for (UIntType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = T(FPBits(v));
      if (isnan(x) || isinf(x))
        continue;

      double res1 = func(x, -x);
      ASSERT_FP_EQ(res1, -x);

      double res2 = func(x, x);
      ASSERT_FP_EQ(res2, x);
    }
  }
};

#define LIST_COPYSIGN_TESTS(T, func)                                           \
  using LlvmLibcCopySignTest = CopySignTest<T>;                                \
  TEST_F(LlvmLibcCopySignTest, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibcCopySignTest, Range) { testRange(&func); }
