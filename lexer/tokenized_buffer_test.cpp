// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lexer/tokenized_buffer.h"

#include <iterator>

#include "diagnostics/diagnostic_emitter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lexer/tokenized_buffer_test_helpers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {
namespace {

using ::Carbon::Testing::ExpectedToken;
using ::Carbon::Testing::HasTokens;
using ::Carbon::Testing::IsKeyValueScalars;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::StrEq;

struct LexerTest : ::testing::Test {
  llvm::SmallVector<SourceBuffer, 16> source_storage;

  auto GetSourceBuffer(llvm::Twine text) -> SourceBuffer& {
    source_storage.push_back(SourceBuffer::CreateFromText(text.str()));
    return source_storage.back();
  }

  auto Lex(llvm::Twine text) -> TokenizedBuffer {
    // TODO: build a full mock for this.
    return TokenizedBuffer::Lex(GetSourceBuffer(text),
                                ConsoleDiagnosticEmitter());
  }
};

TEST_F(LexerTest, HandlesEmptyBuffer) {
  auto buffer = Lex("");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_EQ(buffer.Tokens().begin(), buffer.Tokens().end());
}

TEST_F(LexerTest, TracksLinesAndColumns) {
  auto buffer = Lex("\n  ;;\n   ;;;\n   x\"foo\" \"\"\"baz\n  a\n \"\"\" y");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Semi(),
                           .line = 2,
                           .column = 3,
                           .indent_column = 3},
                          {.kind = TokenKind::Semi(),
                           .line = 2,
                           .column = 4,
                           .indent_column = 3},
                          {.kind = TokenKind::Semi(),
                           .line = 3,
                           .column = 4,
                           .indent_column = 4},
                          {.kind = TokenKind::Semi(),
                           .line = 3,
                           .column = 5,
                           .indent_column = 4},
                          {.kind = TokenKind::Semi(),
                           .line = 3,
                           .column = 6,
                           .indent_column = 4},
                          {.kind = TokenKind::Identifier(),
                           .line = 4,
                           .column = 4,
                           .indent_column = 4,
                           .text = "x"},
                          {.kind = TokenKind::StringLiteral(),
                           .line = 4,
                           .column = 5,
                           .indent_column = 4},
                          {.kind = TokenKind::StringLiteral(),
                           .line = 4,
                           .column = 11,
                           .indent_column = 4},
                          {.kind = TokenKind::Identifier(),
                           .line = 6,
                           .column = 6,
                           .indent_column = 2,
                           .text = "y"},
                      }));
}

TEST_F(LexerTest, HandlesIntegerLiteral) {
  auto buffer = Lex("12-578\n  1  2\n0x12_3ABC\n0b10_10_11\n1_234_567");
  EXPECT_FALSE(buffer.HasErrors());
  ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 1,
                           .column = 1,
                           .indent_column = 1,
                           .text = "12"},
                          {.kind = TokenKind::Minus(),
                           .line = 1,
                           .column = 3,
                           .indent_column = 1},
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 1,
                           .column = 4,
                           .indent_column = 1,
                           .text = "578"},
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 2,
                           .column = 3,
                           .indent_column = 3,
                           .text = "1"},
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 2,
                           .column = 6,
                           .indent_column = 3,
                           .text = "2"},
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 3,
                           .column = 1,
                           .indent_column = 1,
                           .text = "0x12_3ABC"},
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 4,
                           .column = 1,
                           .indent_column = 1,
                           .text = "0b10_10_11"},
                          {.kind = TokenKind::IntegerLiteral(),
                           .line = 5,
                           .column = 1,
                           .indent_column = 1,
                           .text = "1_234_567"},
                      }));
  auto token_12 = buffer.Tokens().begin();
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_12), 12);
  auto token_578 = buffer.Tokens().begin() + 2;
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_578), 578);
  auto token_1 = buffer.Tokens().begin() + 3;
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_1), 1);
  auto token_2 = buffer.Tokens().begin() + 4;
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_2), 2);
  auto token_0x12_3abc = buffer.Tokens().begin() + 5;
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_0x12_3abc), 0x12'3abc);
  auto token_0b10_10_11 = buffer.Tokens().begin() + 6;
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_0b10_10_11), 0b10'10'11);
  auto token_1_234_567 = buffer.Tokens().begin() + 7;
  EXPECT_EQ(buffer.GetIntegerLiteral(*token_1_234_567), 1'234'567);
}

TEST_F(LexerTest, ValidatesBaseSpecifier) {
  llvm::StringLiteral valid[] = {
      // Decimal integer literals.
      "0",
      "1",
      "123456789000000000000000000000000000000000000",

      // Hexadecimal integer literals.
      "0x0123456789ABCDEF",
      "0x0000000000000000000000000000000",

      // Binary integer literals.
      "0b10110100101001010",
      "0b0000000",
  };
  for (llvm::StringLiteral literal : valid) {
    auto buffer = Lex(literal);
    EXPECT_FALSE(buffer.HasErrors()) << literal;
    ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::IntegerLiteral(),
                             .line = 1,
                             .column = 1,
                             .indent_column = 1,
                             .text = literal}}));
  }

  llvm::StringLiteral invalid[] = {
      "00",  "0X123",    "0o123",          "0B1",
      "007", "123L",     "123456789A",     "0x",
      "0b",  "0x123abc", "0b011101201001", "0b10A",
      "0x_", "0b_",
  };
  for (llvm::StringLiteral literal : invalid) {
    auto buffer = Lex(literal);
    EXPECT_TRUE(buffer.HasErrors()) << literal;
    ASSERT_THAT(
        buffer,
        HasTokens(llvm::ArrayRef<ExpectedToken>{{.kind = TokenKind::Error(),
                                                 .line = 1,
                                                 .column = 1,
                                                 .indent_column = 1,
                                                 .text = literal}}));
  }
}

TEST_F(LexerTest, ValidatesIntegerDigitSeparators) {
  llvm::StringLiteral valid[] = {
      // Decimal literals optionally have digit separators every 3 places.
      "1_234",
      "123_456",
      "1_234_567",

      // Hexadecimal literals optionally have digit separators every 4 places.
      "0x1_0000",
      "0x1000_0000",
      "0x1_0000_0000",

      // Binary integer literals can have digit separators anywhere..
      "0b1_0_1_0_1_0",
      "0b111_0000",
  };
  for (llvm::StringLiteral literal : valid) {
    auto buffer = Lex(literal);
    EXPECT_FALSE(buffer.HasErrors()) << literal;
    ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::IntegerLiteral(),
                             .line = 1,
                             .column = 1,
                             .indent_column = 1,
                             .text = literal}}));
  }

  llvm::StringLiteral invalid[] = {
      // Decimal literals.
      "12_34",
      "123_4_6_789",
      "12_3456_789",
      "12__345",
      "1_",

      // Hexadecimal literals.
      "0x_1234",
      "0x123_",
      "0x12_3",
      "0x_234_5678",
      "0x1234_567",

      // Binary literals.
      "0b_10101",
      "0b1__01",
      "0b1011_",
      "0b1_01_01_",
  };
  for (llvm::StringLiteral literal : invalid) {
    auto buffer = Lex(literal);
    EXPECT_TRUE(buffer.HasErrors()) << literal;
    // We expect to produce a token even for a literal containing invalid digit
    // separators, for better error recovery.
    ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::IntegerLiteral(),
                             .line = 1,
                             .column = 1,
                             .indent_column = 1,
                             .text = literal}}));
  }
}

TEST_F(LexerTest, HandlesRealLiteral) {
  struct Testcase {
    llvm::StringLiteral token;
    uint64_t mantissa;
    int64_t exponent;
    unsigned radix;
  };
  Testcase testcases[] = {
      // Decimal real literals.
      {.token = "0.0", .mantissa = 0, .exponent = -1, .radix = 10},
      {.token = "12.345", .mantissa = 12345, .exponent = -3, .radix = 10},
      {.token = "12.345e6", .mantissa = 12345, .exponent = 3, .radix = 10},
      {.token = "12.345e+6", .mantissa = 12345, .exponent = 3, .radix = 10},
      {.token = "1_234.5e-2", .mantissa = 12345, .exponent = -3, .radix = 10},
      {.token = "1.0e-2_000_000",
       .mantissa = 10,
       .exponent = -2'000'001,
       .radix = 10},

      // Hexadecimal real literals.
      {.token = "0x1_2345_6789.CDEF",
       .mantissa = 0x1'2345'6789'CDEF,
       .exponent = -16,
       .radix = 16},
      {.token = "0x0.0001p4", .mantissa = 1, .exponent = -12, .radix = 16},
      {.token = "0x0.0001p+4", .mantissa = 1, .exponent = -12, .radix = 16},
      {.token = "0x0.0001p-4", .mantissa = 1, .exponent = -20, .radix = 16},
      // The exponent here works out as exactly INT64_MIN.
      {.token = "0x1.01p-9223372036854775800",
       .mantissa = 0x101,
       .exponent = -9223372036854775807L - 1L,
       .radix = 16},
      // The exponent here doesn't fit in a signed 64-bit integer until we
      // adjust for the radix point.
      {.token = "0x1.01p9223372036854775809",
       .mantissa = 0x101,
       .exponent = 9223372036854775801L,
       .radix = 16},

      // Binary real literals. These are invalid, but we accept them for error
      // recovery.
      {.token = "0b10_11_01.01",
       .mantissa = 0b10110101,
       .exponent = -2,
       .radix = 2},
  };
  for (Testcase testcase : testcases) {
    auto buffer = Lex(testcase.token);
    EXPECT_EQ(buffer.HasErrors(), testcase.radix == 2);
    ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::RealLiteral(),
                             .line = 1,
                             .column = 1,
                             .indent_column = 1,
                             .text = testcase.token},
                        }));
    auto token = buffer.Tokens().begin();
    TokenizedBuffer::RealLiteralValue value = buffer.GetRealLiteral(*token);
    EXPECT_EQ(value.Mantissa().getZExtValue(), testcase.mantissa);
    EXPECT_EQ(value.Exponent().getSExtValue(), testcase.exponent);
    EXPECT_EQ(value.IsDecimal(), testcase.radix == 10);
  }
}

TEST_F(LexerTest, HandlesRealLiteralOverflow) {
  llvm::StringLiteral input = "0x1.000001p-9223372036854775800";
  auto buffer = Lex(input);
  EXPECT_FALSE(buffer.HasErrors());
  ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::RealLiteral(),
                           .line = 1,
                           .column = 1,
                           .indent_column = 1,
                           .text = input},
                      }));
  auto token = buffer.Tokens().begin();
  TokenizedBuffer::RealLiteralValue value = buffer.GetRealLiteral(*token);
  EXPECT_EQ(value.Mantissa(), 0x1000001);
  EXPECT_EQ((value.Exponent() + 9223372036854775800).getSExtValue(), -24);
  EXPECT_EQ(value.IsDecimal(), false);
}

TEST_F(LexerTest, ValidatesRealLiterals) {
  llvm::StringLiteral invalid_digit_separators[] = {
      // Invalid digit separators.
      "12_34.5",     "123.4_567", "123.456_7", "1_2_3.4",
      "123.4e56_78", "0x12_34.5", "0x12.3_4",  "0x12.34p5_6",
  };
  for (llvm::StringLiteral literal : invalid_digit_separators) {
    auto buffer = Lex(literal);
    EXPECT_TRUE(buffer.HasErrors()) << literal;
    // We expect to produce a token even for a literal containing invalid digit
    // separators, for better error recovery.
    ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::RealLiteral(),
                             .line = 1,
                             .column = 1,
                             .indent_column = 1,
                             .text = literal}}));
  }

  llvm::StringLiteral invalid[] = {
      // No digits in integer part.
      "0x.0",
      "0b.0",
      "0x_.0",
      "0b_.0",

      // No digits in fractional part.
      "0.e",
      "0.e0",
      "0.e+0",
      "0x0.p",
      "0x0.p-0",

      // Invalid digits in mantissa.
      "123A.4",
      "123.4A",
      "123A.4e0",
      "123.4Ae0",
      "0x123ABCDEFG.0",
      "0x123.ABCDEFG",
      "0x123ABCDEFG.0p0",
      "0x123.ABCDEFGp0",

      // Invalid exponent letter.
      "0.0f0",
      "0.0p0",
      "0.0z+0",
      "0x0.0e0",
      "0x0.0f0",
      "0x0.0z-0",

      // No digits in exponent part.
      "0.0e",
      "0x0.0p",
      "0.0e_",
      "0x0.0p_",

      // Invalid digits in exponent part.
      "0.0eHELLO",
      "0.0eA",
      "0.0e+A",
      "0x0.0pA",
      "0x0.0p-A",
  };
  for (llvm::StringLiteral literal : invalid) {
    auto buffer = Lex(literal);
    EXPECT_TRUE(buffer.HasErrors()) << literal;
    ASSERT_THAT(
        buffer,
        HasTokens(llvm::ArrayRef<ExpectedToken>{{.kind = TokenKind::Error(),
                                                 .line = 1,
                                                 .column = 1,
                                                 .indent_column = 1,
                                                 .text = literal}}));
  }
}

TEST_F(LexerTest, SplitsNumericLiteralsProperly) {
  llvm::StringLiteral source_text = R"(
    1.
    .2
    3.+foo
    4.0-bar
    5.0e+123+456
    6.0e+1e+2
    1e7
    8..10
    9.0.9.5
    10.foo
    11.0.foo
    12e+1
    13._
  )";
  auto buffer = Lex(source_text);
  EXPECT_TRUE(buffer.HasErrors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::IntegerLiteral(), .text = "1"},
                  {.kind = TokenKind::Period()},
                  // newline
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::IntegerLiteral(), .text = "2"},
                  // newline
                  {.kind = TokenKind::IntegerLiteral(), .text = "3"},
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::Plus()},
                  {.kind = TokenKind::Identifier(), .text = "foo"},
                  // newline
                  {.kind = TokenKind::RealLiteral(), .text = "4.0"},
                  {.kind = TokenKind::Minus()},
                  {.kind = TokenKind::Identifier(), .text = "bar"},
                  // newline
                  {.kind = TokenKind::RealLiteral(), .text = "5.0e+123"},
                  {.kind = TokenKind::Plus()},
                  {.kind = TokenKind::IntegerLiteral(), .text = "456"},
                  // newline
                  {.kind = TokenKind::Error(), .text = "6.0e+1e"},
                  {.kind = TokenKind::Plus()},
                  {.kind = TokenKind::IntegerLiteral(), .text = "2"},
                  // newline
                  {.kind = TokenKind::Error(), .text = "1e7"},
                  // newline
                  {.kind = TokenKind::IntegerLiteral(), .text = "8"},
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::IntegerLiteral(), .text = "10"},
                  // newline
                  {.kind = TokenKind::RealLiteral(), .text = "9.0"},
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::RealLiteral(), .text = "9.5"},
                  // newline
                  {.kind = TokenKind::Error(), .text = "10.foo"},
                  // newline
                  {.kind = TokenKind::RealLiteral(), .text = "11.0"},
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::Identifier(), .text = "foo"},
                  // newline
                  {.kind = TokenKind::Error(), .text = "12e"},
                  {.kind = TokenKind::Plus()},
                  {.kind = TokenKind::IntegerLiteral(), .text = "1"},
                  // newline
                  {.kind = TokenKind::IntegerLiteral(), .text = "13"},
                  {.kind = TokenKind::Period()},
                  {.kind = TokenKind::UnderscoreKeyword()},
              }));
}

TEST_F(LexerTest, HandlesGarbageCharacters) {
  constexpr char GarbageText[] = "$$💩-$\n$\0$12$\n\"\n\"\\";
  auto buffer = Lex(llvm::StringRef(GarbageText, sizeof(GarbageText) - 1));
  EXPECT_TRUE(buffer.HasErrors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::Error(),
           .line = 1,
           .column = 1,
           .text = llvm::StringRef("$$💩", 6)},
          // 💩 takes 4 bytes, and we count column as bytes offset.
          {.kind = TokenKind::Minus(), .line = 1, .column = 7},
          {.kind = TokenKind::Error(), .line = 1, .column = 8, .text = "$"},
          // newline
          {.kind = TokenKind::Error(),
           .line = 2,
           .column = 1,
           .text = llvm::StringRef("$\0$", 3)},
          {.kind = TokenKind::IntegerLiteral(),
           .line = 2,
           .column = 4,
           .text = "12"},
          {.kind = TokenKind::Error(), .line = 2, .column = 6, .text = "$"},
          // newline
          {.kind = TokenKind::Error(),
           .line = 3,
           .column = 1,
           .text = llvm::StringRef("\"", 1)},
          // newline
          {.kind = TokenKind::Error(),
           .line = 4,
           .column = 1,
           .text = llvm::StringRef("\"", 1)},
          {.kind = TokenKind::Backslash(),
           .line = 4,
           .column = 2,
           .text = llvm::StringRef("\\", 1)},
      }));
}

TEST_F(LexerTest, Symbols) {
  // We don't need to exhaustively test symbols here as they're handled with
  // common code, but we want to check specific patterns to verify things like
  // max-munch rule and handling of interesting symbols.
  auto buffer = Lex("<<<");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::LessLess()},
                          {TokenKind::Less()},
                      }));

  buffer = Lex("<<=>>");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::LessLessEqual()},
                          {TokenKind::GreaterGreater()},
                      }));

  buffer = Lex("< <=> >");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::Less()},
                          {TokenKind::LessEqualGreater()},
                          {TokenKind::Greater()},
                      }));

  buffer = Lex("\\/?@&^!");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::Backslash()},
                          {TokenKind::Slash()},
                          {TokenKind::Question()},
                          {TokenKind::At()},
                          {TokenKind::Amp()},
                          {TokenKind::Caret()},
                          {TokenKind::Exclaim()},
                      }));
}

TEST_F(LexerTest, Parens) {
  auto buffer = Lex("()");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::OpenParen()},
                          {TokenKind::CloseParen()},
                      }));

  buffer = Lex("((()()))");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::OpenParen()},
                          {TokenKind::OpenParen()},
                          {TokenKind::OpenParen()},
                          {TokenKind::CloseParen()},
                          {TokenKind::OpenParen()},
                          {TokenKind::CloseParen()},
                          {TokenKind::CloseParen()},
                          {TokenKind::CloseParen()},
                      }));
}

TEST_F(LexerTest, CurlyBraces) {
  auto buffer = Lex("{}");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::OpenCurlyBrace()},
                          {TokenKind::CloseCurlyBrace()},
                      }));

  buffer = Lex("{{{}{}}}");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::OpenCurlyBrace()},
                          {TokenKind::OpenCurlyBrace()},
                          {TokenKind::OpenCurlyBrace()},
                          {TokenKind::CloseCurlyBrace()},
                          {TokenKind::OpenCurlyBrace()},
                          {TokenKind::CloseCurlyBrace()},
                          {TokenKind::CloseCurlyBrace()},
                          {TokenKind::CloseCurlyBrace()},
                      }));
}

TEST_F(LexerTest, MatchingGroups) {
  {
    TokenizedBuffer buffer = Lex("(){}");
    ASSERT_FALSE(buffer.HasErrors());
    auto it = buffer.Tokens().begin();
    auto open_paren_token = *it++;
    auto close_paren_token = *it++;
    EXPECT_EQ(close_paren_token,
              buffer.GetMatchedClosingToken(open_paren_token));
    EXPECT_EQ(open_paren_token,
              buffer.GetMatchedOpeningToken(close_paren_token));
    auto open_curly_token = *it++;
    auto close_curly_token = *it++;
    EXPECT_EQ(close_curly_token,
              buffer.GetMatchedClosingToken(open_curly_token));
    EXPECT_EQ(open_curly_token,
              buffer.GetMatchedOpeningToken(close_curly_token));
    EXPECT_EQ(buffer.Tokens().end(), it);
  }

  {
    TokenizedBuffer buffer = Lex("({x}){(y)} {{((z))}}");
    ASSERT_FALSE(buffer.HasErrors());
    auto it = buffer.Tokens().begin();
    auto open_paren_token = *it++;
    auto open_curly_token = *it++;
    ASSERT_EQ("x", buffer.GetIdentifierText(buffer.GetIdentifier(*it++)));
    auto close_curly_token = *it++;
    auto close_paren_token = *it++;
    EXPECT_EQ(close_paren_token,
              buffer.GetMatchedClosingToken(open_paren_token));
    EXPECT_EQ(open_paren_token,
              buffer.GetMatchedOpeningToken(close_paren_token));
    EXPECT_EQ(close_curly_token,
              buffer.GetMatchedClosingToken(open_curly_token));
    EXPECT_EQ(open_curly_token,
              buffer.GetMatchedOpeningToken(close_curly_token));

    open_curly_token = *it++;
    open_paren_token = *it++;
    ASSERT_EQ("y", buffer.GetIdentifierText(buffer.GetIdentifier(*it++)));
    close_paren_token = *it++;
    close_curly_token = *it++;
    EXPECT_EQ(close_curly_token,
              buffer.GetMatchedClosingToken(open_curly_token));
    EXPECT_EQ(open_curly_token,
              buffer.GetMatchedOpeningToken(close_curly_token));
    EXPECT_EQ(close_paren_token,
              buffer.GetMatchedClosingToken(open_paren_token));
    EXPECT_EQ(open_paren_token,
              buffer.GetMatchedOpeningToken(close_paren_token));

    open_curly_token = *it++;
    auto inner_open_curly_token = *it++;
    open_paren_token = *it++;
    auto inner_open_paren_token = *it++;
    ASSERT_EQ("z", buffer.GetIdentifierText(buffer.GetIdentifier(*it++)));
    auto inner_close_paren_token = *it++;
    close_paren_token = *it++;
    auto inner_close_curly_token = *it++;
    close_curly_token = *it++;
    EXPECT_EQ(close_curly_token,
              buffer.GetMatchedClosingToken(open_curly_token));
    EXPECT_EQ(open_curly_token,
              buffer.GetMatchedOpeningToken(close_curly_token));
    EXPECT_EQ(inner_close_curly_token,
              buffer.GetMatchedClosingToken(inner_open_curly_token));
    EXPECT_EQ(inner_open_curly_token,
              buffer.GetMatchedOpeningToken(inner_close_curly_token));
    EXPECT_EQ(close_paren_token,
              buffer.GetMatchedClosingToken(open_paren_token));
    EXPECT_EQ(open_paren_token,
              buffer.GetMatchedOpeningToken(close_paren_token));
    EXPECT_EQ(inner_close_paren_token,
              buffer.GetMatchedClosingToken(inner_open_paren_token));
    EXPECT_EQ(inner_open_paren_token,
              buffer.GetMatchedOpeningToken(inner_close_paren_token));

    EXPECT_EQ(buffer.Tokens().end(), it);
  }
}

TEST_F(LexerTest, MismatchedGroups) {
  auto buffer = Lex("{");
  EXPECT_TRUE(buffer.HasErrors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {TokenKind::OpenCurlyBrace()},
                  {.kind = TokenKind::CloseCurlyBrace(), .recovery = true},
              }));

  buffer = Lex("}");
  EXPECT_TRUE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Error(), .text = "}"},
                      }));

  buffer = Lex("{(}");
  EXPECT_TRUE(buffer.HasErrors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::OpenCurlyBrace(), .column = 1},
          {.kind = TokenKind::OpenParen(), .column = 2},
          {.kind = TokenKind::CloseParen(), .column = 3, .recovery = true},
          {.kind = TokenKind::CloseCurlyBrace(), .column = 3},
      }));

  buffer = Lex(")({)");
  EXPECT_TRUE(buffer.HasErrors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::Error(), .column = 1, .text = ")"},
          {.kind = TokenKind::OpenParen(), .column = 2},
          {.kind = TokenKind::OpenCurlyBrace(), .column = 3},
          {.kind = TokenKind::CloseCurlyBrace(), .column = 4, .recovery = true},
          {.kind = TokenKind::CloseParen(), .column = 4},
      }));
}

TEST_F(LexerTest, Keywords) {
  auto buffer = Lex("   fn");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FnKeyword(), .column = 4, .indent_column = 4},
      }));

  buffer = Lex("and or not if else for loop return var break continue _");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::AndKeyword()},
                          {TokenKind::OrKeyword()},
                          {TokenKind::NotKeyword()},
                          {TokenKind::IfKeyword()},
                          {TokenKind::ElseKeyword()},
                          {TokenKind::ForKeyword()},
                          {TokenKind::LoopKeyword()},
                          {TokenKind::ReturnKeyword()},
                          {TokenKind::VarKeyword()},
                          {TokenKind::BreakKeyword()},
                          {TokenKind::ContinueKeyword()},
                          {TokenKind::UnderscoreKeyword()},
                      }));
}

TEST_F(LexerTest, Comments) {
  auto buffer = Lex(" ;\n  // foo\n  ;");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Semi(),
                           .line = 1,
                           .column = 2,
                           .indent_column = 2},
                          {.kind = TokenKind::Semi(),
                           .line = 3,
                           .column = 3,
                           .indent_column = 3},
                      }));

  buffer = Lex("// foo\n//\n// bar");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{}));

  // Make sure weird characters aren't a problem.
  buffer = Lex("  // foo#$!^?@-_💩🍫⃠ [̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅]");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{}));

  // Make sure we can lex a comment at the end of the input.
  buffer = Lex("//");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{}));
}

TEST_F(LexerTest, InvalidComments) {
  llvm::StringLiteral testcases[] = {
      "  /// foo\n",
      "foo // bar\n",
      "//! hello",
      " //world",
  };
  for (llvm::StringLiteral testcase : testcases) {
    auto buffer = Lex(testcase);
    EXPECT_TRUE(buffer.HasErrors());
  }
}

TEST_F(LexerTest, Identifiers) {
  auto buffer = Lex("   foobar");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Identifier(),
                           .column = 4,
                           .indent_column = 4,
                           .text = "foobar"},
                      }));

  // Check different kinds of identifier character sequences.
  buffer = Lex("_foo_bar");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Identifier(), .text = "_foo_bar"},
                      }));

  buffer = Lex("foo2bar00");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::Identifier(), .text = "foo2bar00"},
              }));

  // Check that we can parse identifiers that start with a keyword.
  buffer = Lex("fnord");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Identifier(), .text = "fnord"},
                      }));

  // Check multiple identifiers with indent and interning.
  buffer = Lex("   foo;bar\nbar \n  foo\tfoo");
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::Identifier(),
                           .line = 1,
                           .column = 4,
                           .indent_column = 4,
                           .text = "foo"},
                          {.kind = TokenKind::Semi()},
                          {.kind = TokenKind::Identifier(),
                           .line = 1,
                           .column = 8,
                           .indent_column = 4,
                           .text = "bar"},
                          {.kind = TokenKind::Identifier(),
                           .line = 2,
                           .column = 1,
                           .indent_column = 1,
                           .text = "bar"},
                          {.kind = TokenKind::Identifier(),
                           .line = 3,
                           .column = 3,
                           .indent_column = 3,
                           .text = "foo"},
                          {.kind = TokenKind::Identifier(),
                           .line = 3,
                           .column = 7,
                           .indent_column = 3,
                           .text = "foo"},
                      }));
}

TEST_F(LexerTest, StringLiterals) {
  llvm::StringLiteral testcase = R"(
    "hello world\n"

    """foo
      test \
      \xAB
     """ trailing

      #"""#

    "\0"

    #"\0"foo"\1"#

    """x"""
  )";

  auto buffer = Lex(testcase);
  EXPECT_FALSE(buffer.HasErrors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::StringLiteral(),
                   .line = 2,
                   .column = 5,
                   .indent_column = 5,
                   .string_contents = {"hello world\n"}},
                  {.kind = TokenKind::StringLiteral(),
                   .line = 4,
                   .column = 5,
                   .indent_column = 5,
                   .string_contents = {" test  \xAB\n"}},
                  {.kind = TokenKind::Identifier(),
                   .line = 7,
                   .column = 10,
                   .indent_column = 6,
                   .text = "trailing"},
                  {.kind = TokenKind::StringLiteral(),
                   .line = 9,
                   .column = 7,
                   .indent_column = 7,
                   .string_contents = {"\""}},
                  {.kind = TokenKind::StringLiteral(),
                   .line = 11,
                   .column = 5,
                   .indent_column = 5,
                   .string_contents = llvm::StringLiteral::withInnerNUL("\0")},
                  {.kind = TokenKind::StringLiteral(),
                   .line = 13,
                   .column = 5,
                   .indent_column = 5,
                   .string_contents = {"\\0\"foo\"\\1"}},

                  // """x""" is three string literals, not one.
                  {.kind = TokenKind::StringLiteral(),
                   .line = 15,
                   .column = 5,
                   .indent_column = 5,
                   .string_contents = {""}},
                  {.kind = TokenKind::StringLiteral(),
                   .line = 15,
                   .column = 7,
                   .indent_column = 5,
                   .string_contents = {"x"}},
                  {.kind = TokenKind::StringLiteral(),
                   .line = 15,
                   .column = 10,
                   .indent_column = 5,
                   .string_contents = {""}},
              }));
}

TEST_F(LexerTest, InvalidStringLiterals) {
  llvm::StringLiteral invalid[] = {
      R"(")",
      R"("""
      "")",
      R"("\)",
      R"("\")",
      R"("\\)",
      R"("\\\")",
      R"(""")",
      R"("""
      )",
      R"("""\)",
      R"(#"""
      """)",
  };

  for (llvm::StringLiteral test : invalid) {
    auto buffer = Lex(test);
    EXPECT_TRUE(buffer.HasErrors()) << "`" << test << "`";

    // We should have formed at least one error token.
    bool found_error = false;
    for (TokenizedBuffer::Token token : buffer.Tokens()) {
      if (buffer.GetKind(token) == TokenKind::Error()) {
        found_error = true;
        break;
      }
    }
    EXPECT_TRUE(found_error) << "`" << test << "`";
  }
}

auto GetAndDropLine(llvm::StringRef& text) -> std::string {
  auto newline_offset = text.find_first_of('\n');
  llvm::StringRef line = text.slice(0, newline_offset);

  if (newline_offset != llvm::StringRef::npos) {
    text = text.substr(newline_offset + 1);
  } else {
    text = "";
  }

  return line.str();
}

TEST_F(LexerTest, Printing) {
  auto buffer = Lex(";");
  ASSERT_FALSE(buffer.HasErrors());
  std::string print_storage;
  llvm::raw_string_ostream print_stream(print_storage);
  buffer.Print(print_stream);
  llvm::StringRef print = print_stream.str();
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 0, kind: 'Semi', line: 1, column: 1, "
                    "indent: 1, spelling: ';' }"));
  EXPECT_TRUE(print.empty()) << print;

  // Test kind padding.
  buffer = Lex("(;foo;)");
  ASSERT_FALSE(buffer.HasErrors());
  print_storage.clear();
  buffer.Print(print_stream);
  print = print_stream.str();
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 0, kind:  'OpenParen', line: 1, column: "
                    "1, indent: 1, spelling: '(', closing_token: 4 }"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 1, kind:       'Semi', line: 1, column: "
                    "2, indent: 1, spelling: ';' }"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 2, kind: 'Identifier', line: 1, column: "
                    "3, indent: 1, spelling: 'foo', identifier: 0 }"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 3, kind:       'Semi', line: 1, column: "
                    "6, indent: 1, spelling: ';' }"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 4, kind: 'CloseParen', line: 1, column: "
                    "7, indent: 1, spelling: ')', opening_token: 0 }"));
  EXPECT_TRUE(print.empty()) << print;

  // Test digit padding with max values of 9, 10, and 11.
  buffer = Lex(";\n\n\n\n\n\n\n\n\n\n        ;;");
  ASSERT_FALSE(buffer.HasErrors());
  print_storage.clear();
  buffer.Print(print_stream);
  print = print_stream.str();
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 0, kind: 'Semi', line:  1, column:  1, "
                    "indent: 1, spelling: ';' }"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 1, kind: 'Semi', line: 11, column:  9, "
                    "indent: 9, spelling: ';' }"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("token: { index: 2, kind: 'Semi', line: 11, column: 10, "
                    "indent: 9, spelling: ';' }"));
  EXPECT_TRUE(print.empty()) << print;
}

TEST_F(LexerTest, PrintingAsYaml) {
  // Test that we can parse this into YAML and verify line and indent data.
  auto buffer = Lex("\n ;\n\n\n; ;\n\n\n\n\n\n\n\n\n\n\n");
  ASSERT_FALSE(buffer.HasErrors());
  std::string print_output;
  llvm::raw_string_ostream print_stream(print_output);
  buffer.Print(print_stream);
  print_stream.flush();

  // Parse the output into a YAML stream. This will print errors to stderr.
  llvm::SourceMgr source_manager;
  llvm::yaml::Stream yaml_stream(print_output, source_manager);
  auto yaml_it = yaml_stream.begin();
  auto* root_node = llvm::dyn_cast<llvm::yaml::MappingNode>(yaml_it->getRoot());
  ASSERT_THAT(root_node, NotNull());

  // Walk the top-level mapping of tokens, dig out the sub-mapping of data for
  // each taken, and then verify those entries.
  auto mapping_it = llvm::cast<llvm::yaml::MappingNode>(root_node)->begin();
  auto* token_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*mapping_it);
  ASSERT_THAT(token_node, NotNull());
  auto* token_key_node =
      llvm::dyn_cast<llvm::yaml::ScalarNode>(token_node->getKey());
  ASSERT_THAT(token_key_node, NotNull());
  EXPECT_THAT(token_key_node->getRawValue(), StrEq("token"));
  auto* token_value_node =
      llvm::dyn_cast<llvm::yaml::MappingNode>(token_node->getValue());
  ASSERT_THAT(token_value_node, NotNull());
  auto token_it = token_value_node->begin();
  EXPECT_THAT(&*token_it, IsKeyValueScalars("index", "0"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("kind", "Semi"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("line", "2"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("column", "2"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("indent", "2"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("spelling", ";"));
  EXPECT_THAT(++token_it, Eq(token_value_node->end()));

  ++mapping_it;
  token_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*mapping_it);
  ASSERT_THAT(token_node, NotNull());
  token_key_node = llvm::dyn_cast<llvm::yaml::ScalarNode>(token_node->getKey());
  ASSERT_THAT(token_key_node, NotNull());
  EXPECT_THAT(token_key_node->getRawValue(), StrEq("token"));
  token_value_node =
      llvm::dyn_cast<llvm::yaml::MappingNode>(token_node->getValue());
  ASSERT_THAT(token_value_node, NotNull());
  token_it = token_value_node->begin();
  EXPECT_THAT(&*token_it, IsKeyValueScalars("index", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("kind", "Semi"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("line", "5"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("column", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("indent", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("spelling", ";"));
  EXPECT_THAT(++token_it, Eq(token_value_node->end()));

  ++mapping_it;
  token_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*mapping_it);
  ASSERT_THAT(token_node, NotNull());
  token_key_node = llvm::dyn_cast<llvm::yaml::ScalarNode>(token_node->getKey());
  ASSERT_THAT(token_key_node, NotNull());
  EXPECT_THAT(token_key_node->getRawValue(), StrEq("token"));
  token_value_node =
      llvm::dyn_cast<llvm::yaml::MappingNode>(token_node->getValue());
  ASSERT_THAT(token_value_node, NotNull());
  token_it = token_value_node->begin();
  EXPECT_THAT(&*token_it, IsKeyValueScalars("index", "2"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("kind", "Semi"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("line", "5"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("column", "3"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("indent", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("spelling", ";"));
  EXPECT_THAT(++token_it, Eq(token_value_node->end()));

  ASSERT_THAT(++mapping_it, Eq(root_node->end()));
  ASSERT_THAT(++yaml_it, Eq(yaml_stream.end()));
}

}  // namespace
}  // namespace Carbon
