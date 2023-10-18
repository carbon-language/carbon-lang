// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::Not;

TEST(ValueStore, Integer) {
  CompileValueStores value_stores;
  IntegerId id1 = value_stores.integers().Add(llvm::APInt(64, 1));
  IntegerId id2 = value_stores.integers().Add(llvm::APInt(64, 2));

  ASSERT_TRUE(id1.is_valid());
  ASSERT_TRUE(id2.is_valid());
  EXPECT_THAT(id1, Not(Eq(id2)));

  EXPECT_THAT(value_stores.integers().Get(id1), Eq(1));
  EXPECT_THAT(value_stores.integers().Get(id2), Eq(2));
}

TEST(ValueStore, Real) {
  Real real1{.mantissa = llvm::APInt(64, 1),
             .exponent = llvm::APInt(64, 11),
             .is_decimal = true};
  Real real2{.mantissa = llvm::APInt(64, 2),
             .exponent = llvm::APInt(64, 22),
             .is_decimal = false};

  CompileValueStores value_stores;
  RealId id1 = value_stores.reals().Add(real1);
  RealId id2 = value_stores.reals().Add(real2);

  ASSERT_TRUE(id1.is_valid());
  ASSERT_TRUE(id2.is_valid());
  EXPECT_THAT(id1, Not(Eq(id2)));

  const auto& real1_copy = value_stores.reals().Get(id1);
  EXPECT_THAT(real1.mantissa, Eq(real1_copy.mantissa));
  EXPECT_THAT(real1.exponent, Eq(real1_copy.exponent));
  EXPECT_THAT(real1.is_decimal, Eq(real1_copy.is_decimal));

  const auto& real2_copy = value_stores.reals().Get(id2);
  EXPECT_THAT(real2.mantissa, Eq(real2_copy.mantissa));
  EXPECT_THAT(real2.exponent, Eq(real2_copy.exponent));
  EXPECT_THAT(real2.is_decimal, Eq(real2_copy.is_decimal));
}

TEST(ValueStore, String) {
  std::string a = "a";
  std::string b = "b";
  CompileValueStores value_stores;
  StringId a_id = value_stores.strings().Add(a);
  StringId b_id = value_stores.strings().Add(b);

  ASSERT_TRUE(a_id.is_valid());
  ASSERT_TRUE(b_id.is_valid());

  EXPECT_THAT(a_id, Not(Eq(b_id)));
  EXPECT_THAT(value_stores.strings().Get(a_id), Eq(a));
  EXPECT_THAT(value_stores.strings().Get(b_id), Eq(b));
}

}  // namespace
}  // namespace Carbon
