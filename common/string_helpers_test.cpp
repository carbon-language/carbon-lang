// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/string_helpers.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::Eq;
using ::testing::Optional;

namespace Carbon {
namespace {

TEST(UnescapeStringLiteral, Behavior) {
  EXPECT_THAT(UnescapeStringLiteral("test"), Optional(Eq("test")));
  EXPECT_THAT(UnescapeStringLiteral("test\n"), Optional(Eq("test\n")));
  EXPECT_THAT(UnescapeStringLiteral("test\\n"), Optional(Eq("test\n")));
  EXPECT_THAT(UnescapeStringLiteral("abc\\ndef"), Optional(Eq("abc\ndef")));
  EXPECT_THAT(UnescapeStringLiteral("test\\\\n"), Optional(Eq("test\\n")));
  EXPECT_THAT(UnescapeStringLiteral("\\xAA"), Optional(Eq("\xAA")));
  EXPECT_THAT(UnescapeStringLiteral("\\x12"), Optional(Eq("\x12")));

  // Missing char after `\`.
  EXPECT_THAT(UnescapeStringLiteral("a\\"), Eq(std::nullopt));
  // Not a supported escape.
  EXPECT_THAT(UnescapeStringLiteral("\\e"), Eq(std::nullopt));
  // Needs 2 hex chars.
  EXPECT_THAT(UnescapeStringLiteral("\\x"), Eq(std::nullopt));
  // Needs 2 hex chars.
  EXPECT_THAT(UnescapeStringLiteral("\\xA"), Eq(std::nullopt));
  // Needs uppercase hex.
  EXPECT_THAT(UnescapeStringLiteral("\\xaa"), Eq(std::nullopt));
}

}  // namespace
}  // namespace Carbon
