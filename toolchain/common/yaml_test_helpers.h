// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_COMMON_YAML_TEST_HELPERS_H_
#define TOOLCHAIN_COMMON_YAML_TEST_HELPERS_H_

#include <iomanip>
#include <iostream>
#include <sstream>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Carbon {
namespace Testing {
namespace Yaml {

struct EmptyComparable {
  friend bool operator==(EmptyComparable, EmptyComparable) { return true; }
  friend bool operator!=(EmptyComparable, EmptyComparable) { return false; }
};

struct Value;
struct NullValue : EmptyComparable {};
using ScalarValue = std::string;
using MappingValue = std::vector<std::pair<Value, Value>>;
using SequenceValue = std::vector<Value>;
struct AliasValue : EmptyComparable {};
struct ErrorValue : EmptyComparable {};

// A thin wrapper around a variant of possible YAML value types. This type only
// exists to break the cycle between the variant and MappingValue /
// SequenceValue, and does not represent an abstraction.
struct Value {
  std::variant<NullValue, ScalarValue, MappingValue, SequenceValue, AliasValue,
               ErrorValue>
      value;

  // Forward comparisons to the variant.
  friend bool operator==(const Value& v1, const Value& v2) {
    return v1.value == v2.value;
  }
  template <typename T>
  friend bool operator==(const Value& v1, const T& v2) {
    return v1 == Value{v2};
  }
  template <typename T>
  friend bool operator==(const T& v1, const Value& v2) {
    return Value{v1} == v2;
  }
  friend bool operator!=(const Value& v1, const Value& v2) {
    return v1.value != v2.value;
  }
  template <typename T>
  friend bool operator!=(const Value& v1, const T& v2) {
    return v1 != Value{v2};
  }
  template <typename T>
  friend bool operator!=(const T& v1, const Value& v2) {
    return Value{v1} != v2;
  }

  friend std::ostream& operator<<(std::ostream& os, const Value& v);

  // Parses a sequence of YAML documents from the given YAML file.
  static SequenceValue FromText(llvm::StringRef text);
};

template <typename T>
std::string DescribeMatcher(::testing::Matcher<T> matcher) {
  std::ostringstream out;
  matcher.DescribeTo(&out);
  return out.str();
}

// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Mapping, contents,
          "is mapping that " + DescribeMatcher<MappingValue>(contents)) {
  ::testing::Matcher<MappingValue> contents_matcher = contents;

  if (auto* map = std::get_if<MappingValue>(&arg.value)) {
    return contents_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a mapping";
  return false;
}

// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Sequence, contents,
          "is mapping that " + DescribeMatcher<SequenceValue>(contents)) {
  ::testing::Matcher<SequenceValue> contents_matcher = contents;

  if (auto* map = std::get_if<SequenceValue>(&arg.value)) {
    return contents_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a sequence";
  return false;
}

// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(Scalar, value,
          "has scalar value " + ::testing::PrintToString(value)) {
  ::testing::Matcher<ScalarValue> value_matcher = value;

  if (auto* map = std::get_if<ScalarValue>(&arg.value)) {
    return value_matcher.MatchAndExplain(*map, result_listener);
  }

  *result_listener << "which is not a scalar";
  return false;
}

}  // namespace Yaml
}  // namespace Testing
}  // namespace Carbon

#endif  // TOOLCHAIN_COMMON_YAML_TEST_HELPERS_H_
