// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file provides gmock matchers to support testing YAML output.
//
// A YAML document can be converted into a matchable value using
// Yaml::Value::FromText, and then matched with Yaml::Mapping, Yaml::Sequence,
// or Yaml::Scalar. Scalar values can also be matched directly against strings.
//
// Example usage:
//
//     namespace Yaml = Carbon::Testing::Yaml;
//     using ::testing::ElementsAre;
//     using ::testing::Pair;
//     Yaml::Value yaml = Yaml::Value::FromText(R"yaml(
//     ---
//     fruits:
//       - apple
//       - orange
//       - pear
//     ...
//     ---
//     - [foo: "bar"]: "baz"
//     )yaml"),
//
//     // Exact values can be matched by constructing the desired value.
//     EXPECT_THAT(
//         yaml,
//         ElementsAre(
//             Yaml::MappingValue{
//                 {"fruits", Yaml::SequenceValue{"apple", "orange", "pear"}}},
//             Yaml::SequenceValue{Yaml::MappingValue{
//                 {Yaml::SequenceValue{Yaml::MappingValue{{"foo", "bar"}}},
//                  "baz"}}}));
//
//     // Properties can be checked using Yaml::Mapping or Yaml::Sequence to
//     // adapt regular gmock container matchers.
//     EXPECT_THAT(
//         yaml,
//         Contains(Yaml::Mapping(
//             Contains(Pair("fruits", Yaml::Sequence(Contains("orange")))))));
//
// On match failure, Yaml::Values are printed as C++ code that can be used to
// recreate the value, for easy copy-pasting into test expectations.

#ifndef TOOLCHAIN_COMMON_YAML_TEST_HELPERS_H_
#define TOOLCHAIN_COMMON_YAML_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <variant>

#include "common/ostream.h"

namespace Carbon {
namespace Testing {
namespace Yaml {

struct EmptyComparable {
  friend auto operator==(EmptyComparable, EmptyComparable) -> bool {
    return true;
  }
  friend auto operator!=(EmptyComparable, EmptyComparable) -> bool {
    return false;
  }
};

struct Value;
struct NullValue : EmptyComparable {};
using ScalarValue = std::string;
using MappingValue = std::vector<std::pair<Value, Value>>;
using SequenceValue = std::vector<Value>;
struct AliasValue : EmptyComparable {};
struct ErrorValue : EmptyComparable {};

// A thin wrapper around a variant of possible YAML value types. This type
// intentionally provides no additional encapsulation or invariants beyond
// those of the variant.
struct Value : std::variant<NullValue, ScalarValue, MappingValue, SequenceValue,
                            AliasValue, ErrorValue> {
  using variant::variant;

  // Prints the Value in the form of code to recreate the value.
  friend auto operator<<(std::ostream& os, const Value& v) -> std::ostream&;

  // Parses a sequence of YAML documents from the given YAML text.
  static auto FromText(llvm::StringRef text) -> SequenceValue;
};

template <typename T>
auto DescribeMatcher(::testing::Matcher<T> matcher) -> std::string {
  std::ostringstream out;
  matcher.DescribeTo(&out);
  return out.str();
}

// Match a Value that is a MappingValue.
// Same as testing::VariantWith<MappingValue>(contents).
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Mapping, contents,
          "is mapping that " + DescribeMatcher<MappingValue>(contents)) {
  ::testing::Matcher<MappingValue> contents_matcher = contents;

  if (auto* map = std::get_if<MappingValue>(&arg)) {
    return contents_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a mapping";
  return false;
}

// Match a Value that is a SequenceValue.
// Same as testing::VariantWith<SequenceValue>(contents).
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Sequence, contents,
          "is mapping that " + DescribeMatcher<SequenceValue>(contents)) {
  ::testing::Matcher<SequenceValue> contents_matcher = contents;

  if (auto* map = std::get_if<SequenceValue>(&arg)) {
    return contents_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a sequence";
  return false;
}

// Match a Value that is a ScalarValue.
// Same as testing::VariantWith<ScalarValue>(contents).
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Scalar, value,
          "has scalar value " + ::testing::PrintToString(value)) {
  ::testing::Matcher<ScalarValue> value_matcher = value;

  if (auto* map = std::get_if<ScalarValue>(&arg)) {
    return value_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a scalar";
  return false;
}

}  // namespace Yaml
}  // namespace Testing
}  // namespace Carbon

#endif  // TOOLCHAIN_COMMON_YAML_TEST_HELPERS_H_
