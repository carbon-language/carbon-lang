// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testing/base/test_raw_ostream.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;

TEST(ValueStore, Int) {
  SharedValueStores value_stores;
  IntId id1 = value_stores.ints().Add(llvm::APInt(64, 1));
  IntId id2 = value_stores.ints().Add(llvm::APInt(64, 2));

  ASSERT_TRUE(id1.is_valid());
  ASSERT_TRUE(id2.is_valid());
  EXPECT_THAT(id1, Not(Eq(id2)));

  EXPECT_THAT(value_stores.ints().Get(id1), Eq(1));
  EXPECT_THAT(value_stores.ints().Get(id2), Eq(2));
}

TEST(ValueStore, Real) {
  Real real1{.mantissa = llvm::APInt(64, 1),
             .exponent = llvm::APInt(64, 11),
             .is_decimal = true};
  Real real2{.mantissa = llvm::APInt(64, 2),
             .exponent = llvm::APInt(64, 22),
             .is_decimal = false};

  SharedValueStores value_stores;
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
  SharedValueStores value_stores;
  auto a_id = value_stores.identifiers().Add(a);
  auto b_id = value_stores.string_literal_values().Add(b);

  ASSERT_TRUE(a_id.is_valid());
  ASSERT_TRUE(b_id.is_valid());

  EXPECT_THAT(a_id.index, Not(Eq(b_id.index)));
  EXPECT_THAT(value_stores.identifiers().Get(a_id), Eq(a));
  EXPECT_THAT(value_stores.string_literal_values().Get(b_id), Eq(b));

  // Adding the same string again, even with a different Id type, should return
  // the same id.
  EXPECT_THAT(value_stores.string_literal_values().Add(a).index,
              Eq(a_id.index));
  EXPECT_THAT(value_stores.identifiers().Add(b).index, Eq(b_id.index));
}

auto MatchSharedValues(testing::Matcher<Yaml::MappingValue> ints,
                       testing::Matcher<Yaml::MappingValue> reals,
                       testing::Matcher<Yaml::MappingValue> strings) -> auto {
  return Yaml::IsYaml(Yaml::Sequence(ElementsAre(Yaml::Mapping(ElementsAre(Pair(
      "shared_values",
      Yaml::Mapping(ElementsAre(Pair("ints", Yaml::Mapping(ints)),
                                Pair("reals", Yaml::Mapping(reals)),
                                Pair("strings", Yaml::Mapping(strings))))))))));
}

TEST(ValueStore, PrintEmpty) {
  SharedValueStores value_stores;
  TestRawOstream out;
  value_stores.Print(out);
  EXPECT_THAT(Yaml::Value::FromText(out.TakeStr()),
              MatchSharedValues(IsEmpty(), IsEmpty(), IsEmpty()));
}

TEST(ValueStore, PrintVals) {
  SharedValueStores value_stores;
  llvm::APInt apint(64, 8, /*isSigned=*/true);
  value_stores.ints().Add(apint);
  value_stores.reals().Add(
      Real{.mantissa = apint, .exponent = apint, .is_decimal = true});
  value_stores.string_literal_values().Add("foo'\"baz");
  TestRawOstream out;
  value_stores.Print(out);

  EXPECT_THAT(
      Yaml::Value::FromText(out.TakeStr()),
      MatchSharedValues(ElementsAre(Pair("int0", Yaml::Scalar("8"))),
                        ElementsAre(Pair("real0", Yaml::Scalar("8*10^8"))),
                        ElementsAre(Pair("str0", Yaml::Scalar("foo'\"baz")))));
}

}  // namespace
}  // namespace Carbon::Testing
