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

llvm::Expected<std::string> MakeError(std::string_view message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

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

TEST(ParseBlockStringLiteral, Parse) {
  struct Test {
    std::string source;
    llvm::Expected<std::string> expected;
  };

  std::vector<Test> tests;
  tests.push_back({"", MakeError("Too few lines")});
  tests.push_back({"'a'\n", MakeError("Should start with triple quotes: 'a'")});
  tests.push_back(
      {"\"\"\"carbon file\n",
       MakeError("Invalid characters in file type indicator: carbon file")});
  tests.push_back({"\"\"\"\n", MakeError("Should end with triple quotes: ")});
  tests.push_back(
      {R"("""
     A block string literal
    with wrong indent
     """)",
       MakeError("Wrong indent for line:     with wrong indent, expected 5")});
  tests.push_back({R"("""
     \q
     """)",
                   MakeError("Invalid escaping in \\q")});

  // Empty string.
  tests.push_back({R"("""
""")",
                   ""});

  // One line string.
  tests.push_back({R"("""
     A block string literal
     """)",
                   R"(A block string literal
)"});

  // Two line string.
  tests.push_back({R"("""
     A block string literal
       with indent.
     """)",
                   R"(A block string literal
  with indent.
)"});

  // With file type indicator.
  tests.push_back({R"("""carbon
     A block string literal
       with file type indicator.
     """)",
                   R"(A block string literal
  with file type indicator.
)"});

  // Whitespace after opening quotes.
  tests.push_back({R"("""
     A block string literal
     """)",
                   R"(A block string literal
)"});

  // With empty lines.
  tests.push_back({R"("""
     A block string literal

       with

       empty

       lines.
     """)",
                   R"(A block string literal

  with

  empty

  lines.
)"});

  // With \<newline> escape.
  tests.push_back({R"("""
     A block string literal\
     """)",
                   "A block string literal"});
  tests.push_back({R"("""
     A block string literal\\
     """)",
                   R"(A block string literal\
)"});
  tests.push_back({R"("""
     A block string literal\\\
     """)",
                   R"(A block string literal\)"});
  tests.push_back({R"("""
     A block string literal\
     \
     \
     \
     """)",
                   "A block string literal"});

  for (Test& test : tests) {
    SCOPED_TRACE(test.source);
    llvm::Expected<std::string> parsed = ParseBlockStringLiteral(test.source);
    if (!test.expected) {
      const std::string expected_error = toString(test.expected.takeError());
      ASSERT_FALSE(!!parsed) << "Expected error: " << expected_error;
      EXPECT_EQ(expected_error, toString(parsed.takeError()));
    } else {
      ASSERT_TRUE(!!parsed)
          << "Unexpected error: " << toString(parsed.takeError());
      EXPECT_EQ(*test.expected, *parsed);
    }
  }
}

}  // namespace
}  // namespace Carbon
