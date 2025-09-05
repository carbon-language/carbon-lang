// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "toolchain/base/value_ids.h"

namespace Carbon::Testing {
namespace {

using ::testing::Eq;
using ::testing::Not;

TEST(ValueStore, Real) {
  Real real1{.mantissa = llvm::APInt(64, 1),
             .exponent = llvm::APInt(64, 11),
             .is_decimal = true};
  Real real2{.mantissa = llvm::APInt(64, 2),
             .exponent = llvm::APInt(64, 22),
             .is_decimal = false};

  ValueStore<RealId, Real> reals;
  RealId id1 = reals.Add(real1);
  RealId id2 = reals.Add(real2);

  ASSERT_TRUE(id1.has_value());
  ASSERT_TRUE(id2.has_value());
  EXPECT_THAT(id1, Not(Eq(id2)));

  const auto& real1_copy = reals.Get(id1);
  EXPECT_THAT(real1.mantissa, Eq(real1_copy.mantissa));
  EXPECT_THAT(real1.exponent, Eq(real1_copy.exponent));
  EXPECT_THAT(real1.is_decimal, Eq(real1_copy.is_decimal));

  const auto& real2_copy = reals.Get(id2);
  EXPECT_THAT(real2.mantissa, Eq(real2_copy.mantissa));
  EXPECT_THAT(real2.exponent, Eq(real2_copy.exponent));
  EXPECT_THAT(real2.is_decimal, Eq(real2_copy.is_decimal));
}

}  // namespace
}  // namespace Carbon::Testing
