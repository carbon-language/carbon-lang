// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/template_string.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon {
namespace {

using ::testing::StrEq;

template <TemplateString S>
constexpr auto FromTemplate() -> llvm::StringRef {
  return S;
}

template <TemplateString S>
constexpr auto CStrFromTemplate() -> const char* {
  return S.c_str();
}

template <TemplateString>
constexpr auto IsValidTemplateString(int) -> std::true_type { return {}; }

struct AnythingAsTemplateArg {
  template <typename T>
  constexpr AnythingAsTemplateArg(T&&) {}
};
template <AnythingAsTemplateArg>
constexpr auto IsValidTemplateString(...) -> std::false_type { return {}; }

// Compile time tests with `static_assert`
static_assert(FromTemplate<"test">().size() == 4,
              "Not usable in a `constexpr` context.");
static_assert(__builtin_strlen(CStrFromTemplate<"test">()) == 4,
              "Not usable in a `constexpr` context.");

// The string must not contain embedded nulls.
static_assert(IsValidTemplateString<"test">(0));
static_assert(!IsValidTemplateString<"te\0st">(0));

// The string must be null-terminated.
using FourChars = char[4];
static_assert(IsValidTemplateString<FourChars{'t', 'e', 's', 0}>(0));
static_assert(!IsValidTemplateString<FourChars{'t', 'e', 's', 't'}>(0));

TEST(TemplateStringTest, Test) {
  EXPECT_THAT(FromTemplate<"test">(), StrEq("test"));
  EXPECT_THAT(CStrFromTemplate<"test">(), StrEq("test"));

  constexpr char GoodStr[5] = {'t', 'e', 's', 't', '\0'};
  static_assert(IsValidTemplateString<GoodStr>(0));
  EXPECT_THAT(FromTemplate<GoodStr>(), StrEq("test"));

  constexpr char BadStr[4] = {'t', 'e', 's', 't'};
  static_assert(!IsValidTemplateString<BadStr>(0));
}

}  // namespace
}  // namespace Carbon
