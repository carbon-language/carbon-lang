// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/string_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "llvm/Support/Error.h"

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

TEST(ParseBlockStringLiteral, FailTooFewLines) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral("");
  ASSERT_FALSE(!!parsed);
  EXPECT_EQ("Too few lines", toString(parsed.takeError()));
}

TEST(ParseBlockStringLiteral, FailNoLeadingTripleQuotes) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral("'a'\n");
  ASSERT_FALSE(!!parsed);
  EXPECT_EQ("Should start with triple quotes: 'a'",
            toString(parsed.takeError()));
}

TEST(ParseBlockStringLiteral, FailInvalideFiletypeIndicator) {
  llvm::Expected<std::string> parsed =
      ParseBlockStringLiteral("\"\"\"carbon file\n");
  ASSERT_FALSE(!!parsed);
  EXPECT_EQ("Invalid characters in file type indicator: carbon file",
            toString(parsed.takeError()));
}

TEST(ParseBlockStringLiteral, FailEndingTripleQuotes) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral("\"\"\"\n");
  ASSERT_FALSE(!!parsed);
  EXPECT_EQ("Should end with triple quotes: ", toString(parsed.takeError()));
}

TEST(ParseBlockStringLiteral, FailWrongIndent) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal
    with wrong indent
     """)");
  ASSERT_FALSE(!!parsed);
  EXPECT_EQ("Wrong indent for line:     with wrong indent, expected 5",
            toString(parsed.takeError()));
}

TEST(ParseBlockStringLiteral, FailInvalidEscaping) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     \q
     """)");
  ASSERT_FALSE(!!parsed);
  EXPECT_EQ("Invalid escaping in \\q", toString(parsed.takeError()));
}

TEST(ParseBlockStringLiteral, OkEmptyString) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
""")");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ("", *parsed);
}

TEST(ParseBlockStringLiteral, OkOneLineString) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal
)",
            *parsed);
}

TEST(ParseBlockStringLiteral, OkTwoLineString) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal
       with indent.
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal
  with indent.
)",
            *parsed);
}

TEST(ParseBlockStringLiteral, OkWithFileTypeIndicator) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""carbon
     A block string literal
       with file type indicator.
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal
  with file type indicator.
)",
            *parsed);
}

TEST(ParseBlockStringLiteral, OkWhitespaceAfterOpeningQuotes) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal
)",
            *parsed);
}

TEST(ParseBlockStringLiteral, OkWithEmptyLines) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal

       with

       empty

       lines.
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal

  with

  empty

  lines.
)",
            *parsed);
}

TEST(ParseBlockStringLiteral, OkWithSlashNewlineEscape) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal\
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ("A block string literal", *parsed);
}

TEST(ParseBlockStringLiteral, OkWithDoubleSlashNewline) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal\\
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal\
)",
            *parsed);
}

TEST(ParseBlockStringLiteral, OkWithTripleSlashNewline) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal\\\
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ(R"(A block string literal\)", *parsed);
}

TEST(ParseBlockStringLiteral, OkMultipleSlashes) {
  llvm::Expected<std::string> parsed = ParseBlockStringLiteral(R"("""
     A block string literal\
     \
     \
     \
     """)");
  ASSERT_TRUE(!!parsed) << "Unexpected error: " << toString(parsed.takeError());
  EXPECT_EQ("A block string literal", *parsed);
}

}  // namespace
}  // namespace Carbon
