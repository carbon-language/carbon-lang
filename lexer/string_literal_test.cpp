// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lexer/string_literal.h"

#include "diagnostics/diagnostic_emitter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lexer/test_helpers.h"

namespace Carbon {
namespace {

struct StringLiteralTest : ::testing::Test {
  StringLiteralTest() : error_tracker(ConsoleDiagnosticConsumer()) {}

  ErrorTrackingDiagnosticConsumer error_tracker;

  auto Lex(llvm::StringRef text) -> LexedStringLiteral {
    llvm::Optional<LexedStringLiteral> result = LexedStringLiteral::Lex(text);
    assert(result);
    EXPECT_EQ(result->Text(), text);
    return *result;
  }

  auto Parse(llvm::StringRef text) -> std::string {
    LexedStringLiteral token = Lex(text);
    Testing::SingleTokenDiagnosticTranslator translator(text);
    DiagnosticEmitter<const char*> emitter(translator, error_tracker);
    return token.ComputeValue(emitter);
  }
};

TEST_F(StringLiteralTest, StringLiteralBounds) {
  llvm::StringLiteral valid[] = {
      R"("")",
      R"("""
      """)",
      R"("""
      "foo"
      """)",

      // Escaped terminators don't end the string.
      R"("\"")",
      R"("\\")",
      R"("\\\"")",
      R"("""
      \"""
      """)",
      R"("""
      "\""
      """)",
      R"("""
      ""\"
      """)",
      R"("""
      ""\
      """)",
      R"(#"""
      """\#n
      """#)",

      // Only a matching number of '#'s terminates the string.
      R"(#""#)",
      R"(#"xyz"foo"#)",
      R"(##"xyz"#foo"##)",
      R"(#"\""#)",

      // Escape sequences likewise require a matching number of '#'s.
      R"(#"\#"#"#)",
      R"(#"\"#)",
      R"(#"""
      \#"""#
      """#)",

      // #"""# does not start a multiline string literal.
      R"(#"""#)",
      R"(##"""##)",
  };

  for (llvm::StringLiteral test : valid) {
    llvm::Optional<LexedStringLiteral> result = LexedStringLiteral::Lex(test);
    EXPECT_TRUE(result.hasValue()) << test;
    if (result) {
      EXPECT_EQ(result->Text(), test);
    }
  }

  llvm::StringLiteral invalid[] = {
      R"(")",
      R"("""
      "")",
      R"("\)",  //
      R"("\")",
      R"("\\)",  //
      R"("\\\")",
      R"("""
      )",
      R"(#"""
      """)",
      R"(" \
      ")",
  };

  for (llvm::StringLiteral test : invalid) {
    EXPECT_FALSE(LexedStringLiteral::Lex(test).hasValue())
        << "`" << test << "`";
  }
}

TEST_F(StringLiteralTest, StringLiteralContents) {
  // We use ""s strings to handle embedded nul characters below.
  using std::operator""s;

  std::pair<llvm::StringLiteral, llvm::StringLiteral> testcases[] = {
      // Empty strings.
      {R"("")", ""},

      {R"(
"""
"""
       )",
       ""},

      // Nearly-empty strings.
      {R"(
"""

"""
       )",
       "\n"},

      // Indent removal.
      {R"(
       """file type indicator
          indented contents \
         """
       )",
       " indented contents "},

      {R"(
    """
   hello
  world

   end of test
  """
       )",
       " hello\nworld\n\n end of test\n"},

      // Escape sequences.
      {R"(
       "\x14,\u{1234},\u{00000010},\n,\r,\t,\0,\",\',\\"
       )",
       llvm::StringLiteral::withInnerNUL(
           "\x14,\xE1\x88\xB4,\x10,\x0A,\x0D,\x09,\x00,\x22,\x27,\x5C")},

      {R"(
       "\0A\x1234"
       )",
       llvm::StringLiteral::withInnerNUL("\0A\x12"
                                         "34")},

      {R"(
       "\u{D7FF},\u{E000},\u{10FFFF}"
       )",
       "\xED\x9F\xBF,\xEE\x80\x80,\xF4\x8F\xBF\xBF"},

      // Escape sequences in 'raw' strings.
      {R"(
       #"\#x00,\#xFF,\#u{56789},\#u{ABCD},\#u{00000000000000000EF}"#
       )",
       llvm::StringLiteral::withInnerNUL(
           "\x00,\xFF,\xF1\x96\x9E\x89,\xEA\xAF\x8D,\xC3\xAF")},

      {R"(
       ##"\n,\#n,\##n,\##\##n,\##\###n"##
       )",
       "\\n,\\#n,\n,\\##n,\\###n"},

      // Trailing whitespace handling.
      {"\"\"\"\n  Hello \\\n  World \t \n  Bye!  \\\n  \"\"\"",
       "Hello World\nBye!  "},
  };

  for (auto [test, contents] : testcases) {
    error_tracker.Reset();
    auto value = Parse(test.trim());
    EXPECT_FALSE(error_tracker.SeenError()) << "`" << test << "`";
    EXPECT_EQ(value, contents);
  }
}

TEST_F(StringLiteralTest, StringLiteralBadIndent) {
  std::pair<llvm::StringLiteral, llvm::StringLiteral> testcases[] = {
      // Indent doesn't match the last line.
      {"\"\"\"\n \tx\n  \"\"\"", "x\n"},
      {"\"\"\"\n x\n  \"\"\"", "x\n"},
      {"\"\"\"\n  x\n\t\"\"\"", "x\n"},
      {"\"\"\"\n  ok\n bad\n  \"\"\"", "ok\nbad\n"},
      {"\"\"\"\n bad\n  ok\n  \"\"\"", "bad\nok\n"},
      {"\"\"\"\n  escaped,\\\n bad\n  \"\"\"", "escaped,bad\n"},

      // Indent on last line is followed by text.
      {"\"\"\"\n  x\n  x\"\"\"", "x\nx"},
      {"\"\"\"\n   x\n  x\"\"\"", " x\nx"},
      {"\"\"\"\n x\n  x\"\"\"", "x\nx"},
  };

  for (auto [test, contents] : testcases) {
    error_tracker.Reset();
    auto value = Parse(test);
    EXPECT_TRUE(error_tracker.SeenError()) << "`" << test << "`";
    EXPECT_EQ(value, contents);
  }
}

TEST_F(StringLiteralTest, StringLiteralBadEscapeSequence) {
  llvm::StringLiteral testcases[] = {
      R"("\a")",
      R"("\b")",
      R"("\e")",
      R"("\f")",
      R"("\v")",
      R"("\?")",
      R"("\1")",
      R"("\9")",

      // \0 can't be followed by a decimal digit.
      R"("\01")",
      R"("\09")",

      // \x requires two (uppercase) hexadecimal digits.
      R"("\x")",
      R"("\x0")",
      R"("\x0G")",
      R"("\xab")",
      R"("\x\n")",
      R"("\x\"")",

      // \u requires a braced list of one or more hexadecimal digits.
      R"("\u")",
      R"("\u?")",
      R"("\u\"")",
      R"("\u{")",
      R"("\u{}")",
      R"("\u{A")",
      R"("\u{G}")",
      R"("\u{0000012323127z}")",
      R"("\u{-3}")",

      // \u must specify a non-surrogate code point.
      R"("\u{110000}")",
      R"("\u{000000000000000000000000000000000110000}")",
      R"("\u{D800}")",
      R"("\u{DFFF}")",
  };

  for (llvm::StringLiteral test : testcases) {
    error_tracker.Reset();
    auto value = Parse(test);
    EXPECT_TRUE(error_tracker.SeenError()) << "`" << test << "`";
    // TODO: Test value produced by error recovery.
  }
}

}  // namespace
}  // namespace Carbon
