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

// Compile time tests with `static_assert`
static_assert(FromTemplate<"test">().size() == 4,
              "Not usable in a `constexpr` context.");
static_assert(__builtin_strlen(CStrFromTemplate<"test">()) == 4,
              "Not usable in a `constexpr` context.");

TEST(TemplateStringTest, Test) {
  EXPECT_THAT(FromTemplate<"test">(), StrEq("test"));
  EXPECT_THAT(CStrFromTemplate<"test">(), StrEq("test"));

  // Uncomment to test compile-time rejection of embedded `0`-bytes.
  // EXPECT_THAT(FromTemplate<"test\0test">(), StrEq("test"));

  constexpr char GoodStr[5] = {'t', 'e', 's', 't', '\0'};
  EXPECT_THAT(FromTemplate<GoodStr>(), StrEq("test"));

  // Uncomment to test compile-time rejection of missing null terminator.
  // constexpr char BadStr[4] = {'t', 'e', 's', 't'};
  // EXPECT_THAT(FromTemplate<BadStr>(), StrEq("test"));
}

}  // namespace
}  // namespace Carbon
