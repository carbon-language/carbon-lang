// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/string_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

using ::testing::Eq;
using ::testing::Optional;

namespace Carbon {
namespace {

TEST(UnescapeStringLiteral, Valid) {
  EXPECT_THAT(UnescapeStringLiteral("test"), Optional(Eq("test")));
  EXPECT_THAT(UnescapeStringLiteral("okay whitespace"),
              Optional(Eq("okay whitespace")));
  EXPECT_THAT(UnescapeStringLiteral("test\n"), Optional(Eq("test\n")));
  EXPECT_THAT(UnescapeStringLiteral("test\\n"), Optional(Eq("test\n")));
  EXPECT_THAT(UnescapeStringLiteral("abc\\ndef"), Optional(Eq("abc\ndef")));
  EXPECT_THAT(UnescapeStringLiteral("test\\\\n"), Optional(Eq("test\\n")));
  EXPECT_THAT(UnescapeStringLiteral("\\xAA"), Optional(Eq("\xAA")));
  EXPECT_THAT(UnescapeStringLiteral("\\x12"), Optional(Eq("\x12")));
}

TEST(UnescapeStringLiteral, Invalid) {
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
  // Reserved.
  EXPECT_THAT(UnescapeStringLiteral("\\00"), Eq(std::nullopt));
}

TEST(UnescapeStringLiteral, Nul) {
  std::optional<std::string> str = UnescapeStringLiteral("a\\0b");
  ASSERT_NE(str, std::nullopt);
  EXPECT_THAT(str->size(), Eq(3));
  EXPECT_THAT(strlen(str->c_str()), Eq(1));
  EXPECT_THAT((*str)[0], Eq('a'));
  EXPECT_THAT((*str)[1], Eq('\0'));
  EXPECT_THAT((*str)[2], Eq('b'));
}

}  // namespace
}  // namespace Carbon
