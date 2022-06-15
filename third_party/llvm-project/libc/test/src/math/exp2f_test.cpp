//===-- Unittests for exp2f -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp2f.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcExp2fTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::exp2f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, __llvm_libc::exp2f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, __llvm_libc::exp2f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::exp2f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::exp2f(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcExp2fTest, Overflow) {
  errno = 0;
  EXPECT_FP_EQ(inf, __llvm_libc::exp2f(float(FPBits(0x7f7fffffU))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::exp2f(float(FPBits(0x43000000U))));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(inf, __llvm_libc::exp2f(float(FPBits(0x43000001U))));
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST(LlvmLibcExp2fTest, TrickyInputs) {
  constexpr int N = 12;
  constexpr uint32_t INPUTS[N] = {
      0x3b429d37U, /*0x1.853a6ep-9f*/
      0x3c02a9adU, /*0x1.05535ap-7f*/
      0x3ca66e26U, /*0x1.4cdc4cp-6f*/
      0x3d92a282U, /*0x1.254504p-4f*/
      0x42fa0001U, /*0x1.f40002p+6f*/
      0x42ffffffU, /*0x1.fffffep+6f*/
      0xb8d3d026U, /*-0x1.a7a04cp-14f*/
      0xbcf3a937U, /*-0x1.e7526ep-6f*/
      0xc2fa0001U, /*-0x1.f40002p+6f*/
      0xc2fc0000U, /*-0x1.f8p+6f*/
      0xc2fc0001U, /*-0x1.f80002p+6f*/
      0xc3150000U, /*-0x1.2ap+7f*/
  };
  for (int i = 0; i < N; ++i) {
    errno = 0;
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                   __llvm_libc::exp2f(x), 0.5);
    EXPECT_MATH_ERRNO(0);
  }
}

TEST(LlvmLibcExp2fTest, Underflow) {
  errno = 0;
  EXPECT_FP_EQ(0.0f, __llvm_libc::exp2f(float(FPBits(0xff7fffffU))));
  EXPECT_MATH_ERRNO(ERANGE);

  float x = float(FPBits(0xc3158000U));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                 __llvm_libc::exp2f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = float(FPBits(0xc3160000U));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                 __llvm_libc::exp2f(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);

  x = float(FPBits(0xc3165432U));
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                 __llvm_libc::exp2f(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST(LlvmLibcExp2fTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::exp2f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                   __llvm_libc::exp2f(x), 0.5);
  }
}
