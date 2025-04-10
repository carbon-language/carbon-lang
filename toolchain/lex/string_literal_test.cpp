// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/string_literal.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/check.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/test_helpers.h"

namespace Carbon::Lex {
namespace {

class StringLiteralTest : public ::testing::Test {
 public:
  StringLiteralTest() : error_tracker_(Diagnostics::ConsoleConsumer()) {}

  auto Lex(llvm::StringRef text) -> StringLiteral {
    std::optional<StringLiteral> result = StringLiteral::Lex(text);
    CARBON_CHECK(result);
    EXPECT_EQ(result->text(), text);
    return *result;
  }

  auto Parse(llvm::StringRef text) -> llvm::StringRef {
    StringLiteral token = Lex(text);
    Testing::SingleTokenDiagnosticEmitter emitter(&error_tracker_, text);
    return token.ComputeValue(allocator_, emitter);
  }

  llvm::BumpPtrAllocator allocator_;
  Diagnostics::ErrorTrackingConsumer error_tracker_;
};

TEST_F(StringLiteralTest, StringLiteralBounds) {
  llvm::StringLiteral valid[] = {
      R"("")",
      R"('''
      ''')",
      R"('''
      "foo"
      ''')",

      // Lex """-delimited block string literals for error recovery.
      R"("""
      """)",
      R"("""
      "foo"
      """)",

      // Escaped terminators don't end the string.
      R"("\"")",
      R"("\\")",
      R"("\\\"")",
      R"('''
      \'''
      ''')",
      R"('''
      '\''
      ''')",
      R"('''
      ''\'
      ''')",
      R"('''
      ''\
      ''')",
      R"(#'''
      '''\#n
      '''#)",

      // Only a matching number of '#'s terminates the string.
      R"(#""#)",
      R"(#"xyz"foo"#)",
      R"(##"xyz"#foo"##)",
      R"(#"\""#)",

      // Escape sequences likewise require a matching number of '#'s.
      R"(#"\#"#"#)",
      R"(#"\"#)",
      R"(#'''
      \#'''#
      '''#)",

      // #"""# does not start a multiline string literal.
      R"(#"""#)",
      R"(##"""##)",
  };

  for (llvm::StringLiteral test : valid) {
    SCOPED_TRACE(test);
    std::optional<StringLiteral> result = StringLiteral::Lex(test);
    EXPECT_TRUE(result.has_value());
    if (result) {
      EXPECT_EQ(result->text(), test);
    }
  }

  llvm::StringLiteral invalid[] = {
      // clang-format off
      R"(")",
      R"("\)",
      R"("\")",
      R"("\\)",
      R"("\\\")",
      "'''\n",
      "'''\n'",
      "'''\n''",
      "#'''\n'''",
      R"(" \
      ")",
      // clang-format on
  };

  for (llvm::StringLiteral test : invalid) {
    SCOPED_TRACE(test);
    std::optional<StringLiteral> result = StringLiteral::Lex(test);
    EXPECT_TRUE(result.has_value());
    if (result) {
      EXPECT_FALSE(result->is_terminated());
    }
  }
}

TEST_F(StringLiteralTest, StringLiteralContents) {
  std::pair<llvm::StringLiteral, llvm::StringLiteral> testcases[] = {
      // Empty strings.
      {R"("")", ""},

      {R"(
'''
'''
       )",
       ""},

      // Nearly-empty strings.
      {R"(
'''

'''
       )",
       "\n"},

      // Lines containing only whitespace are treated as empty even if they
      // contain tabs.
      {"'''\n\t  \t\n'''", "\n"},

      // Indent removal.
      {R"(
       '''file type indicator
          indented contents \
         '''
       )",
       " indented contents "},

      // Removal of tabs in indent and suffix.
      {"'''\n \t  hello \t \n \t '''", " hello\n"},

      {R"(
    '''
   hello
  world

   end of test
  '''
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
      {"'''\n  Hello \\\n  World \t \n  Bye!  \\\n  '''",
       "Hello World\nBye!  "},
      {"'''\n\\t\n'''", "\t\n"},
      {"'''\n\\t \n'''", "\t\n"},
  };

  for (auto [test, expected] : testcases) {
    error_tracker_.Reset();
    auto value = Parse(test.trim());
    EXPECT_FALSE(error_tracker_.seen_error()) << "`" << test << "`";
    EXPECT_EQ(value, expected);
  }
}

TEST_F(StringLiteralTest, DoubleQuotedMultiLineLiteral) {
  // For error recovery, """-delimited literals are lexed, but rejected.
  std::pair<llvm::StringLiteral, llvm::StringLiteral> testcases[] = {
      {R"(
"""
'''
"""
       )",
       "'''\n"},
      {R"(
#"""
\#tx
"""#
       )",
       "\tx\n"},
      {R"(
"""abcxyz
   hello\
   """
       )",
       "hello"},
  };

  for (auto [test, contents] : testcases) {
    error_tracker_.Reset();
    auto value = Parse(test.trim());
    EXPECT_TRUE(error_tracker_.seen_error()) << "`" << test << "`";
    EXPECT_EQ(value, contents);
  }
}

TEST_F(StringLiteralTest, StringLiteralBadIndent) {
  std::pair<llvm::StringLiteral, llvm::StringLiteral> testcases[] = {
      // Indent doesn't match the last line.
      {"'''\n \tx\n  '''", "x\n"},
      {"'''\n x\n  '''", "x\n"},
      {"'''\n  x\n\t'''", "x\n"},
      {"'''\n  ok\n bad\n  '''", "ok\nbad\n"},
      {"'''\n bad\n  ok\n  '''", "bad\nok\n"},
      {"'''\n  escaped,\\\n bad\n  '''", "escaped,bad\n"},

      // Indent on last line is followed by text.
      {"'''\n  x\n  x'''", "x\nx"},
      {"'''\n   x\n  x'''", " x\nx"},
      {"'''\n x\n  x'''", "x\nx"},
  };

  for (auto [test, contents] : testcases) {
    error_tracker_.Reset();
    auto value = Parse(test);
    EXPECT_TRUE(error_tracker_.seen_error()) << "`" << test << "`";
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
    error_tracker_.Reset();
    Parse(test);
    EXPECT_TRUE(error_tracker_.seen_error()) << "`" << test << "`";
    // TODO: Test value produced by error recovery.
  }
}

TEST_F(StringLiteralTest, TabInString) {
  auto value = Parse("\"x\ty\"");
  EXPECT_TRUE(error_tracker_.seen_error());
  EXPECT_EQ(value, "x\ty");
}

TEST_F(StringLiteralTest, TabAtEndOfString) {
  auto value = Parse("\"\t\t\t\"");
  EXPECT_TRUE(error_tracker_.seen_error());
  EXPECT_EQ(value, "\t\t\t");
}

TEST_F(StringLiteralTest, TabInBlockString) {
  auto value = Parse("'''\nx\ty\n'''");
  EXPECT_TRUE(error_tracker_.seen_error());
  EXPECT_EQ(value, "x\ty\n");
}

TEST_F(StringLiteralTest, UnicodeTooManyDigits) {
  std::string text = "u{";
  text.append(10000, '9');
  text.append("}");
  auto value = Parse("\"\\" + text + "\"");
  EXPECT_TRUE(error_tracker_.seen_error());
  EXPECT_EQ(value, text);
}

}  // namespace
}  // namespace Carbon::Lex
