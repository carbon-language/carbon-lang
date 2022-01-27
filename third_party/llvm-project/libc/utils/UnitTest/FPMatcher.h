//===-- TestMatchers.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H

#include "src/__support/FPUtil/FPBits.h"

#include "utils/UnitTest/Test.h"

namespace __llvm_libc {
namespace fputil {
namespace testing {

template <typename ValType>
cpp::EnableIfType<cpp::IsFloatingPointType<ValType>::Value, void>
describeValue(const char *label, ValType value,
              testutils::StreamWrapper &stream);

template <typename T, __llvm_libc::testing::TestCondition Condition>
class FPMatcher : public __llvm_libc::testing::Matcher<T> {
  static_assert(__llvm_libc::cpp::IsFloatingPointType<T>::Value,
                "FPMatcher can only be used with floating point values.");
  static_assert(Condition == __llvm_libc::testing::Cond_EQ ||
                    Condition == __llvm_libc::testing::Cond_NE,
                "Unsupported FPMathcer test condition.");

  T expected;
  T actual;

public:
  FPMatcher(T expectedValue) : expected(expectedValue) {}

  bool match(T actualValue) {
    actual = actualValue;
    fputil::FPBits<T> actualBits(actual), expectedBits(expected);
    if (Condition == __llvm_libc::testing::Cond_EQ)
      return (actualBits.is_nan() && expectedBits.is_nan()) ||
             (actualBits.uintval() == expectedBits.uintval());

    // If condition == Cond_NE.
    if (actualBits.is_nan())
      return !expectedBits.is_nan();
    return expectedBits.is_nan() ||
           (actualBits.uintval() != expectedBits.uintval());
  }

  void explainError(testutils::StreamWrapper &stream) override {
    describeValue("Expected floating point value: ", expected, stream);
    describeValue("  Actual floating point value: ", actual, stream);
  }
};

template <__llvm_libc::testing::TestCondition C, typename T>
FPMatcher<T, C> getMatcher(T expectedValue) {
  return FPMatcher<T, C>(expectedValue);
}

} // namespace testing
} // namespace fputil
} // namespace __llvm_libc

#define DECLARE_SPECIAL_CONSTANTS(T)                                           \
  using FPBits = __llvm_libc::fputil::FPBits<T>;                               \
  using UIntType = typename FPBits::UIntType;                                  \
  const T zero = T(FPBits::zero());                                            \
  const T neg_zero = T(FPBits::neg_zero());                                    \
  const T aNaN = T(FPBits::build_nan(1));                                      \
  const T inf = T(FPBits::inf());                                              \
  const T neg_inf = T(FPBits::neg_inf());

#define EXPECT_FP_EQ(expected, actual)                                         \
  EXPECT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_EQ>( \
          expected))

#define ASSERT_FP_EQ(expected, actual)                                         \
  ASSERT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_EQ>( \
          expected))

#define EXPECT_FP_NE(expected, actual)                                         \
  EXPECT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_NE>( \
          expected))

#define ASSERT_FP_NE(expected, actual)                                         \
  ASSERT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_NE>( \
          expected))

#endif // LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H
