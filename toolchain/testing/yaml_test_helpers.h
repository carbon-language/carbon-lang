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
//         Yaml::IsYaml(ElementsAre(
//             Yaml::MappingValue{
//                 {"fruits", Yaml::SequenceValue{"apple", "orange", "pear"}}},
//             Yaml::SequenceValue{Yaml::MappingValue{
//                 {Yaml::SequenceValue{Yaml::MappingValue{{"foo", "bar"}}},
//                  "baz"}}})));
//
//     // Properties can be checked using Yaml::Mapping or Yaml::Sequence to
//     // adapt regular gmock container matchers.
//     EXPECT_THAT(
//         yaml,
//         Yaml::IsYaml(Contains(Yaml::Mapping(
//             Contains(Pair("fruits", Yaml::Sequence(Contains("orange"))))))));
//
// On match failure, Yaml::Values are printed as C++ code that can be used to
// recreate the value, for easy copy-pasting into test expectations.

#ifndef CARBON_TOOLCHAIN_TESTING_YAML_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_TESTING_YAML_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <variant>

#include "common/error.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Testing::Yaml {

struct EmptyComparable {
  friend auto operator==(EmptyComparable /*lhs*/, EmptyComparable /*rhs*/)
      -> bool {
    return true;
  }
  friend auto operator!=(EmptyComparable /*lhs*/, EmptyComparable /*rhs*/)
      -> bool {
    return false;
  }
};

struct Value;
struct NullValue : EmptyComparable {};
using ScalarValue = std::string;
using MappingValue = std::vector<std::pair<Value, Value>>;
using SequenceValue = std::vector<Value>;
struct AliasValue : EmptyComparable {};

// A thin wrapper around a variant of possible YAML value types. This type
// intentionally provides no additional encapsulation or invariants beyond
// those of the variant.
struct Value : std::variant<NullValue, ScalarValue, MappingValue, SequenceValue,
                            AliasValue> {
  using variant::variant;

  // Prints the Value in the form of code to recreate the value.
  friend auto operator<<(std::ostream& os, const Value& v) -> std::ostream&;

  // Parses a sequence of YAML documents from the given YAML text.
  static auto FromText(llvm::StringRef text) -> ErrorOr<SequenceValue>;
};

// Used to examine the results of Value::FromText.
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(IsYaml, matcher,
          "is yaml root sequence that " +
              ::testing::DescribeMatcher<SequenceValue>(matcher)) {
  const ErrorOr<SequenceValue>& yaml = arg;
  const ::testing::Matcher<SequenceValue>& typed_matcher = matcher;
  if (yaml.ok()) {
    // It's hard to intercept printing of the ErrorOr value, so just print it
    // here.
    *result_listener << "\n  which is: " << *yaml << "\n  ";
    return typed_matcher.MatchAndExplain(*yaml, result_listener);
  }

  *result_listener << "\n  with the error: " << yaml.error() << "\n  ";
  return false;
}

// Match a Value that is a MappingValue.
// Similar to testing::VariantWith<MappingValue>(matcher), but with better
// descriptions.
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Mapping, matcher,
          "is mapping that " +
              ::testing::DescribeMatcher<MappingValue>(matcher)) {
  const Value& val = arg;
  const ::testing::Matcher<MappingValue>& typed_matcher = matcher;
  if (const auto* map = std::get_if<MappingValue>(&val)) {
    return typed_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a mapping";
  return false;
}

// Match a Value that is a SequenceValue.
// Similar to testing::VariantWith<SequenceValue>(matcher), but with better
// descriptions.
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Sequence, matcher,
          "is sequence that " +
              ::testing::DescribeMatcher<SequenceValue>(matcher)) {
  const Value& val = arg;
  const ::testing::Matcher<SequenceValue>& typed_matcher = matcher;
  if (const auto* map = std::get_if<SequenceValue>(&val)) {
    return typed_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a sequence";
  return false;
}

// Match a Value that is a ScalarValue.
// Similar to testing::VariantWith<ScalarValue>(matcher), but with better
// descriptions.
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Scalar, matcher,
          "has scalar value " +
              ::testing::DescribeMatcher<ScalarValue>(matcher)) {
  const Value& val = arg;
  const ::testing::Matcher<ScalarValue>& typed_matcher = matcher;
  if (const auto* map = std::get_if<ScalarValue>(&val)) {
    return typed_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a scalar";
  return false;
}

}  // namespace Carbon::Testing::Yaml

#endif  // CARBON_TOOLCHAIN_TESTING_YAML_TEST_HELPERS_H_
