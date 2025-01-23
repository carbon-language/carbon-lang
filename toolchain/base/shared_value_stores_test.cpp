// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/shared_value_stores.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/raw_string_ostream.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

auto MatchSharedValues(testing::Matcher<Yaml::MappingValue> ints,
                       testing::Matcher<Yaml::MappingValue> reals,
                       testing::Matcher<Yaml::MappingValue> identifiers,
                       testing::Matcher<Yaml::MappingValue> strings) -> auto {
  return Yaml::IsYaml(Yaml::Sequence(ElementsAre(Yaml::Mapping(ElementsAre(Pair(
      "shared_values",
      Yaml::Mapping(ElementsAre(Pair("ints", Yaml::Mapping(ints)),
                                Pair("reals", Yaml::Mapping(reals)),
                                Pair("identifiers", Yaml::Mapping(identifiers)),
                                Pair("strings", Yaml::Mapping(strings))))))))));
}

TEST(SharedValueStores, PrintEmpty) {
  SharedValueStores value_stores;
  RawStringOstream out;
  value_stores.Print(out);
  EXPECT_THAT(Yaml::Value::FromText(out.TakeStr()),
              MatchSharedValues(IsEmpty(), IsEmpty(), IsEmpty(), IsEmpty()));
}

TEST(SharedValueStores, PrintVals) {
  SharedValueStores value_stores;
  llvm::APInt apint(64, 8, /*isSigned=*/true);
  value_stores.ints().AddSigned(apint);
  value_stores.ints().AddSigned(llvm::APInt(64, 999'999'999'999));
  value_stores.reals().Add(
      Real{.mantissa = apint, .exponent = apint, .is_decimal = true});
  value_stores.identifiers().Add("a");
  value_stores.string_literal_values().Add("foo'\"baz");
  RawStringOstream out;
  value_stores.Print(out);

  EXPECT_THAT(Yaml::Value::FromText(out.TakeStr()),
              MatchSharedValues(
                  ElementsAre(Pair("ap_int0", Yaml::Scalar("999999999999"))),
                  ElementsAre(Pair("real0", Yaml::Scalar("8*10^8"))),
                  ElementsAre(Pair("identifier0", Yaml::Scalar("a"))),
                  ElementsAre(Pair("string0", Yaml::Scalar("foo'\"baz")))));
}

}  // namespace
}  // namespace Carbon::Testing
