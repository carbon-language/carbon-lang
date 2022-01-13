//===-- Utility class to test ceil[f|l] -------------------------*- C++ -*-===//
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

template <typename T> class CeilTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CeilFunc)(T);

  void testSpecialNumbers(CeilFunc func) {
    EXPECT_FP_EQ(zero, func(zero));
    EXPECT_FP_EQ(neg_zero, func(neg_zero));

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf));

    EXPECT_FP_EQ(aNaN, func(aNaN));
  }

  void testRoundedNumbers(CeilFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(1.0)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.0)));
    EXPECT_FP_EQ(T(10.0), func(T(10.0)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.0)));
    EXPECT_FP_EQ(T(1234.0), func(T(1234.0)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.0)));
  }

  void testFractions(CeilFunc func) {
    EXPECT_FP_EQ(T(1.0), func(T(0.5)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.5)));
    EXPECT_FP_EQ(T(1.0), func(T(0.115)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.115)));
    EXPECT_FP_EQ(T(1.0), func(T(0.715)));
    EXPECT_FP_EQ(T(-0.0), func(T(-0.715)));
    EXPECT_FP_EQ(T(2.0), func(T(1.3)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.3)));
    EXPECT_FP_EQ(T(2.0), func(T(1.5)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.5)));
    EXPECT_FP_EQ(T(2.0), func(T(1.75)));
    EXPECT_FP_EQ(T(-1.0), func(T(-1.75)));
    EXPECT_FP_EQ(T(11.0), func(T(10.32)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.32)));
    EXPECT_FP_EQ(T(11.0), func(T(10.65)));
    EXPECT_FP_EQ(T(-10.0), func(T(-10.65)));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.38)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.38)));
    EXPECT_FP_EQ(T(1235.0), func(T(1234.96)));
    EXPECT_FP_EQ(T(-1234.0), func(T(-1234.96)));
  }

  void testRange(CeilFunc func) {
    constexpr UIntType COUNT = 10000000;
    constexpr UIntType STEP = UIntType(-1) / COUNT;
    for (UIntType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = T(FPBits(v));
      if (isnan(x) || isinf(x))
        continue;

      ASSERT_MPFR_MATCH(mpfr::Operation::Ceil, x, func(x), 0.0);
    }
  }
};

#define LIST_CEIL_TESTS(T, func)                                               \
  using LlvmLibcCeilTest = CeilTest<T>;                                        \
  TEST_F(LlvmLibcCeilTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  TEST_F(LlvmLibcCeilTest, RoundedNubmers) { testRoundedNumbers(&func); }      \
  TEST_F(LlvmLibcCeilTest, Fractions) { testFractions(&func); }                \
  TEST_F(LlvmLibcCeilTest, Range) { testRange(&func); }
