// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/template_string.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>

namespace Carbon {
namespace {

using ::testing::StrEq;

template <TemplateString S>
constexpr auto TemplateAsStringRef() -> llvm::StringRef {
  return S;
}

template <TemplateString S>
constexpr auto TemplateAsStringLiteral() -> llvm::StringLiteral {
  return S;
}

template <TemplateString S>
constexpr auto TemplateAsCStr() -> const char* {
  return S.c_str();
}

// An overload that will be active when it is passed a valid `TemplateString`.
// Returns a true type to allow detection of a valid `TemplateString` argument.
template <TemplateString /*Unused*/>
constexpr auto IsValidTemplateString(int /*unused*/) -> std::true_type {
  return {};
}

// A struct that can be used as a template parameter for any template argument.
struct AnythingAsTemplateArg {
  // An implicit constructor that can accept any argument and discards it.
  template <typename T>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  explicit(false) constexpr AnythingAsTemplateArg(T&& /*unused*/) {}
};

// An overload that will be active for any template argument. Returns a false
// type and is used to detect when a template argument cannot correctly match a
// `TemplateString`.
template <AnythingAsTemplateArg /*Unused*/>
constexpr auto IsValidTemplateString(...) -> std::false_type {
  return {};
}

// Compile time tests with `static_assert`
static_assert(TemplateAsStringRef<"test">().size() == 4,
              "Not usable in a `constexpr` context.");
static_assert(TemplateAsStringLiteral<"test">().size() == 4,
              "Not usable in a `constexpr` context.");
static_assert(__builtin_strlen(TemplateAsCStr<"test">()) == 4,
              "Not usable in a `constexpr` context.");

// The string must not contain embedded nulls.
static_assert(IsValidTemplateString<"test">(0));
static_assert(!IsValidTemplateString<"test\0test">(0));

// The string must be null-terminated.
using FourChars = char[4];
static_assert(IsValidTemplateString<FourChars{'t', 'e', 's', 0}>(0));
static_assert(!IsValidTemplateString<FourChars{'t', 'e', 's', 't'}>(0));

TEST(TemplateStringTest, Test) {
  EXPECT_THAT(TemplateAsStringRef<"test">(), StrEq("test"));
  EXPECT_THAT(TemplateAsStringLiteral<"test">(), StrEq("test"));
  EXPECT_THAT(TemplateAsCStr<"test">(), StrEq("test"));

  constexpr char GoodStr[5] = {'t', 'e', 's', 't', '\0'};
  static_assert(IsValidTemplateString<GoodStr>(0));
  EXPECT_THAT(TemplateAsStringRef<GoodStr>(), StrEq("test"));
  EXPECT_THAT(TemplateAsStringLiteral<GoodStr>(), StrEq("test"));

  constexpr char BadStr[4] = {'t', 'e', 's', 't'};
  static_assert(!IsValidTemplateString<BadStr>(0));
}

}  // namespace
}  // namespace Carbon
