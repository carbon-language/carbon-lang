// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/string_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "llvm/Support/Error.h"

using ::llvm::toString;
using ::testing::Eq;
using ::testing::Optional;

namespace Carbon::Testing {
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
  EXPECT_THAT(UnescapeStringLiteral("test", 1), Optional(Eq("test")));
  EXPECT_THAT(UnescapeStringLiteral("test\\#n", 1), Optional(Eq("test\n")));
  EXPECT_THAT(UnescapeStringLiteral(
                  "r\\u{000000E9}al \\u{2764}\\u{FE0F}\\u{1F50A}!\\u{10FFFF}"),
              Optional(Eq("réal ❤️🔊!􏿿")));
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
  EXPECT_THAT(UnescapeStringLiteral("\\#00", 1), Eq(std::nullopt));
}

TEST(UnescapeStringLiteral, InvalidUnicodes) {
  // Various incomplete Unicode specifiers
  EXPECT_THAT(UnescapeStringLiteral("\\u"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\u1"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\uz"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\u{"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\u{z"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\u{E9"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\u{E9z"), Eq(std::nullopt));
  EXPECT_THAT(UnescapeStringLiteral("\\u{}"), Eq(std::nullopt));

  // invalid characters in unicode
  EXPECT_THAT(UnescapeStringLiteral("\\u{z}"), Eq(std::nullopt));

  // lowercase hexadecimal
  EXPECT_THAT(UnescapeStringLiteral("\\u{e9}"), Eq(std::nullopt));

  // Codepoint number too high
  EXPECT_THAT(UnescapeStringLiteral("\\u{110000}"), Eq(std::nullopt));

  // codepoint more than 8 hex digits
  EXPECT_THAT(UnescapeStringLiteral("\\u{FF000000E9}"), Eq(std::nullopt));
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
  EXPECT_THAT(ParseBlockStringLiteral("").error().message(),
              Eq("Too few lines"));
}

TEST(ParseBlockStringLiteral, FailNoLeadingTripleQuotes) {
  EXPECT_THAT(ParseBlockStringLiteral("'a'\n").error().message(),
              Eq("Should start with triple quotes: 'a'"));
}

TEST(ParseBlockStringLiteral, FailInvalideFiletypeIndicator) {
  EXPECT_THAT(ParseBlockStringLiteral("\"\"\"carbon file\n").error().message(),
              Eq("Invalid characters in file type indicator: carbon file"));
}

TEST(ParseBlockStringLiteral, FailEndingTripleQuotes) {
  EXPECT_THAT(ParseBlockStringLiteral("\"\"\"\n").error().message(),
              Eq("Should end with triple quotes: "));
}

TEST(ParseBlockStringLiteral, FailWrongIndent) {
  constexpr char Input[] = R"("""
     A block string literal
    with wrong indent
     """)";
  EXPECT_THAT(ParseBlockStringLiteral(Input).error().message(),
              Eq("Wrong indent for line:     with wrong indent, expected 5"));
}

TEST(ParseBlockStringLiteral, FailInvalidEscaping) {
  constexpr char Input[] = R"("""
     \q
     """)";
  EXPECT_THAT(ParseBlockStringLiteral(Input).error().message(),
              Eq("Invalid escaping in \\q"));
  constexpr char InputRaw[] = R"("""
     \#q
     """)";
  EXPECT_THAT(ParseBlockStringLiteral(InputRaw, 1).error().message(),
              Eq("Invalid escaping in \\#q"));
}

TEST(ParseBlockStringLiteral, OkEmptyString) {
  constexpr char Input[] = R"("""
""")";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(""));
}

TEST(ParseBlockStringLiteral, OkOneLineString) {
  constexpr char Input[] = R"("""
     A block string literal
     """)";
  constexpr char Expected[] = R"(A block string literal
)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkTwoLineString) {
  constexpr char Input[] = R"("""
     A block string literal
       with indent.
     """)";
  constexpr char Expected[] = R"(A block string literal
  with indent.
)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkWithFileTypeIndicator) {
  constexpr char Input[] = R"("""carbon
     A block string literal
       with file type indicator.
     """)";
  constexpr char Expected[] = R"(A block string literal
  with file type indicator.
)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkWhitespaceAfterOpeningQuotes) {
  constexpr char Input[] = R"("""
     A block string literal
     """)";
  constexpr char Expected[] = R"(A block string literal
)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkWithEmptyLines) {
  constexpr char Input[] = R"("""
     A block string literal

       with

       empty

       lines.
     """)";
  constexpr char Expected[] = R"(A block string literal

  with

  empty

  lines.
)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkWithSlashNewlineEscape) {
  constexpr char Input[] = R"("""
     A block string literal\
     """)";
  constexpr char Expected[] = "A block string literal";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkWithDoubleSlashNewline) {
  constexpr char Input[] = R"("""
     A block string literal\\
     """)";
  constexpr char Expected[] = R"(A block string literal\
)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkWithTripleSlashNewline) {
  constexpr char Input[] = R"("""
     A block string literal\\\
     """)";
  constexpr char Expected[] = R"(A block string literal\)";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

TEST(ParseBlockStringLiteral, OkMultipleSlashes) {
  constexpr char Input[] = R"("""
     A block string literal\
     \
     \
     \
     """)";
  constexpr char Expected[] = "A block string literal";
  EXPECT_THAT(*ParseBlockStringLiteral(Input), Eq(Expected));
}

}  // namespace
}  // namespace Carbon::Testing
