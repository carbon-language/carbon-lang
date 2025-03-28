// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <forward_list>
#include <iterator>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer_test_helpers.h"
#include "toolchain/testing/compile_helper.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::Lex {
namespace {

using ::Carbon::Testing::ExpectedToken;
using ::Carbon::Testing::IsSingleDiagnostic;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Pair;

namespace Yaml = ::Carbon::Testing::Yaml;

class LexerTest : public ::testing::Test {
 public:
  Testing::CompileHelper compile_helper_;
};

TEST_F(LexerTest, HandlesEmptyBuffer) {
  auto& buffer = compile_helper_.GetTokenizedBuffer("");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::FileStart},
                          {.kind = TokenKind::FileEnd}}));
}

TEST_F(LexerTest, TracksLinesAndColumns) {
  auto& buffer = compile_helper_.GetTokenizedBuffer(
      "\n  ;;\n   ;;;\n   x\"foo\" '''baz\n  a\n ''' y");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart,
           .line = 1,
           .column = 1,
           .indent_column = 1},
          {.kind = TokenKind::Semi, .line = 2, .column = 3, .indent_column = 3},
          {.kind = TokenKind::Semi, .line = 2, .column = 4, .indent_column = 3},
          {.kind = TokenKind::Semi, .line = 3, .column = 4, .indent_column = 4},
          {.kind = TokenKind::Semi, .line = 3, .column = 5, .indent_column = 4},
          {.kind = TokenKind::Semi, .line = 3, .column = 6, .indent_column = 4},
          {.kind = TokenKind::Identifier,
           .line = 4,
           .column = 4,
           .indent_column = 4,
           .text = "x"},
          {.kind = TokenKind::StringLiteral,
           .line = 4,
           .column = 5,
           .indent_column = 4},
          {.kind = TokenKind::StringLiteral,
           .line = 4,
           .column = 11,
           .indent_column = 4},
          {.kind = TokenKind::Identifier,
           .line = 6,
           .column = 6,
           .indent_column = 11,
           .text = "y"},
          {.kind = TokenKind::FileEnd, .line = 6, .column = 7},
      }));
}

TEST_F(LexerTest, TracksLinesAndColumnsCrLf) {
  auto& buffer = compile_helper_.GetTokenizedBuffer(
      "\r\n  ;;\r\n   ;;;\r\n   x\"foo\" '''baz\r\n  a\r\n ''' y");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart,
           .line = 1,
           .column = 1,
           .indent_column = 1},
          {.kind = TokenKind::Semi, .line = 2, .column = 3, .indent_column = 3},
          {.kind = TokenKind::Semi, .line = 2, .column = 4, .indent_column = 3},
          {.kind = TokenKind::Semi, .line = 3, .column = 4, .indent_column = 4},
          {.kind = TokenKind::Semi, .line = 3, .column = 5, .indent_column = 4},
          {.kind = TokenKind::Semi, .line = 3, .column = 6, .indent_column = 4},
          {.kind = TokenKind::Identifier,
           .line = 4,
           .column = 4,
           .indent_column = 4,
           .text = "x"},
          {.kind = TokenKind::StringLiteral,
           .line = 4,
           .column = 5,
           .indent_column = 4},
          {.kind = TokenKind::StringLiteral,
           .line = 4,
           .column = 11,
           .indent_column = 4},
          {.kind = TokenKind::Identifier,
           .line = 6,
           .column = 6,
           .indent_column = 11,
           .text = "y"},
          {.kind = TokenKind::FileEnd, .line = 6, .column = 7},
      }));
}

TEST_F(LexerTest, InvalidCR) {
  auto& buffer = compile_helper_.GetTokenizedBuffer("\n ;;\r ;\n   x");
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart,
           .line = 1,
           .column = 1,
           .indent_column = 1},
          {.kind = TokenKind::Semi, .line = 2, .column = 2, .indent_column = 2},
          {.kind = TokenKind::Semi, .line = 2, .column = 3, .indent_column = 2},
          {.kind = TokenKind::Semi, .line = 2, .column = 6, .indent_column = 2},
          {.kind = TokenKind::Identifier,
           .line = 3,
           .column = 4,
           .indent_column = 4,
           .text = "x"},
          {.kind = TokenKind::FileEnd, .line = 3, .column = 5},
      }));
}

TEST_F(LexerTest, InvalidLfCr) {
  auto& buffer = compile_helper_.GetTokenizedBuffer("\n ;;\n\r ;\n   x");
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart,
           .line = 1,
           .column = 1,
           .indent_column = 1},
          {.kind = TokenKind::Semi, .line = 2, .column = 2, .indent_column = 2},
          {.kind = TokenKind::Semi, .line = 2, .column = 3, .indent_column = 2},
          {.kind = TokenKind::Semi, .line = 3, .column = 3, .indent_column = 1},
          {.kind = TokenKind::Identifier,
           .line = 4,
           .column = 4,
           .indent_column = 4,
           .text = "x"},
          {.kind = TokenKind::FileEnd, .line = 4, .column = 5},
      }));
}

TEST_F(LexerTest, HandlesNumericLiteral) {
  auto [buffer, value_stores] =
      compile_helper_.GetTokenizedBufferWithSharedValueStore(
          "12-578\n  1  2\n0x12_3ABC\n0b10_10_11\n1_234_567\n1.5e9");
  EXPECT_FALSE(buffer.has_errors());
  ASSERT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},
                  {.kind = TokenKind::IntLiteral,
                   .line = 1,
                   .column = 1,
                   .indent_column = 1,
                   .text = "12"},
                  {.kind = TokenKind::Minus,
                   .line = 1,
                   .column = 3,
                   .indent_column = 1},
                  {.kind = TokenKind::IntLiteral,
                   .line = 1,
                   .column = 4,
                   .indent_column = 1,
                   .text = "578"},
                  {.kind = TokenKind::IntLiteral,
                   .line = 2,
                   .column = 3,
                   .indent_column = 3,
                   .text = "1"},
                  {.kind = TokenKind::IntLiteral,
                   .line = 2,
                   .column = 6,
                   .indent_column = 3,
                   .text = "2"},
                  {.kind = TokenKind::IntLiteral,
                   .line = 3,
                   .column = 1,
                   .indent_column = 1,
                   .text = "0x12_3ABC"},
                  {.kind = TokenKind::IntLiteral,
                   .line = 4,
                   .column = 1,
                   .indent_column = 1,
                   .text = "0b10_10_11"},
                  {.kind = TokenKind::IntLiteral,
                   .line = 5,
                   .column = 1,
                   .indent_column = 1,
                   .text = "1_234_567"},
                  {.kind = TokenKind::RealLiteral,
                   .line = 6,
                   .column = 1,
                   .indent_column = 1,
                   .text = "1.5e9"},
                  {.kind = TokenKind::FileEnd, .line = 6, .column = 6},
              }));
  auto token_start = buffer.tokens().begin();
  auto token_12 = token_start + 1;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_12)), 12);
  auto token_578 = token_12 + 2;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_578)), 578);
  auto token_1 = token_578 + 1;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_1)), 1);
  auto token_2 = token_1 + 1;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_2)), 2);
  auto token_0x12_3abc = token_2 + 1;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_0x12_3abc)),
            0x12'3abc);
  auto token_0b10_10_11 = token_0x12_3abc + 1;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_0b10_10_11)),
            0b10'10'11);
  auto token_1_234_567 = token_0b10_10_11 + 1;
  EXPECT_EQ(value_stores.ints().Get(buffer.GetIntLiteral(*token_1_234_567)),
            1'234'567);
  auto token_1_5e9 = token_1_234_567 + 1;
  auto value_1_5e9 =
      value_stores.reals().Get(buffer.GetRealLiteral(*token_1_5e9));
  EXPECT_EQ(value_1_5e9.mantissa.getZExtValue(), 15);
  EXPECT_EQ(value_1_5e9.exponent.getSExtValue(), 8);
  EXPECT_EQ(value_1_5e9.is_decimal, true);
}

TEST_F(LexerTest, HandlesInvalidNumericLiterals) {
  auto& buffer =
      compile_helper_.GetTokenizedBuffer("14x 15_49 0x3.5q 0x3_4.5_6 0ops");
  EXPECT_TRUE(buffer.has_errors());
  ASSERT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},
                  {.kind = TokenKind::Error,
                   .line = 1,
                   .column = 1,
                   .indent_column = 1,
                   .text = "14x"},
                  {.kind = TokenKind::IntLiteral,
                   .line = 1,
                   .column = 5,
                   .indent_column = 1,
                   .text = "15_49"},
                  {.kind = TokenKind::Error,
                   .line = 1,
                   .column = 11,
                   .indent_column = 1,
                   .text = "0x3.5q"},
                  {.kind = TokenKind::RealLiteral,
                   .line = 1,
                   .column = 18,
                   .indent_column = 1,
                   .text = "0x3_4.5_6"},
                  {.kind = TokenKind::Error,
                   .line = 1,
                   .column = 28,
                   .indent_column = 1,
                   .text = "0ops"},
                  {.kind = TokenKind::FileEnd, .line = 1, .column = 32},
              }));
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
  auto& buffer = compile_helper_.GetTokenizedBuffer(source_text);
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::FileStart},
                          {.kind = TokenKind::IntLiteral, .text = "1"},
                          {.kind = TokenKind::Period},
                          // newline
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::IntLiteral, .text = "2"},
                          // newline
                          {.kind = TokenKind::IntLiteral, .text = "3"},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::Plus},
                          {.kind = TokenKind::Identifier, .text = "foo"},
                          // newline
                          {.kind = TokenKind::RealLiteral, .text = "4.0"},
                          {.kind = TokenKind::Minus},
                          {.kind = TokenKind::Identifier, .text = "bar"},
                          // newline
                          {.kind = TokenKind::RealLiteral, .text = "5.0e+123"},
                          {.kind = TokenKind::Plus},
                          {.kind = TokenKind::IntLiteral, .text = "456"},
                          // newline
                          {.kind = TokenKind::Error, .text = "6.0e+1e"},
                          {.kind = TokenKind::Plus},
                          {.kind = TokenKind::IntLiteral, .text = "2"},
                          // newline
                          {.kind = TokenKind::Error, .text = "1e7"},
                          // newline
                          {.kind = TokenKind::IntLiteral, .text = "8"},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::IntLiteral, .text = "10"},
                          // newline
                          {.kind = TokenKind::RealLiteral, .text = "9.0"},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::IntLiteral, .text = "9"},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::IntLiteral, .text = "5"},
                          // newline
                          {.kind = TokenKind::Error, .text = "10.foo"},
                          // newline
                          {.kind = TokenKind::RealLiteral, .text = "11.0"},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::Identifier, .text = "foo"},
                          // newline
                          {.kind = TokenKind::Error, .text = "12e"},
                          {.kind = TokenKind::Plus},
                          {.kind = TokenKind::IntLiteral, .text = "1"},
                          // newline
                          {.kind = TokenKind::IntLiteral, .text = "13"},
                          {.kind = TokenKind::Period},
                          {.kind = TokenKind::Underscore},
                          // newline
                          {.kind = TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, HandlesGarbageCharacters) {
  constexpr char GarbageText[] = "$$💩-$\n$\0$12$\n\\\"\\\n\"x";
  auto& buffer = compile_helper_.GetTokenizedBuffer(
      llvm::StringRef(GarbageText, sizeof(GarbageText) - 1));
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart, .line = 1, .column = 1},
          {.kind = TokenKind::Error,
           .line = 1,
           .column = 1,
           // 💩 takes 4 bytes, and we count column as bytes offset.
           .text = llvm::StringRef("$$💩", 6)},
          {.kind = TokenKind::Minus, .line = 1, .column = 7},
          {.kind = TokenKind::Error, .line = 1, .column = 8, .text = "$"},
          // newline
          {.kind = TokenKind::Error,
           .line = 2,
           .column = 1,
           .text = llvm::StringRef("$\0$", 3)},
          {.kind = TokenKind::IntLiteral, .line = 2, .column = 4, .text = "12"},
          {.kind = TokenKind::Error, .line = 2, .column = 6, .text = "$"},
          // newline
          {.kind = TokenKind::Backslash, .line = 3, .column = 1, .text = "\\"},
          {.kind = TokenKind::Error, .line = 3, .column = 2, .text = "\"\\"},
          // newline
          {.kind = TokenKind::Error, .line = 4, .column = 1, .text = "\"x"},
          {.kind = TokenKind::FileEnd, .line = 4, .column = 3},
      }));
}

TEST_F(LexerTest, Symbols) {
  // We don't need to exhaustively test symbols here as they're handled with
  // common code, but we want to check specific patterns to verify things like
  // max-munch rule and handling of interesting symbols.
  auto& buffer1 = compile_helper_.GetTokenizedBuffer("<<<");
  EXPECT_FALSE(buffer1.has_errors());
  EXPECT_THAT(buffer1, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::LessLess},
                           {.kind = TokenKind::Less},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer2 = compile_helper_.GetTokenizedBuffer("<<=>>");
  EXPECT_FALSE(buffer2.has_errors());
  EXPECT_THAT(buffer2, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::LessLessEqual},
                           {.kind = TokenKind::GreaterGreater},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer3 = compile_helper_.GetTokenizedBuffer("< <=> >");
  EXPECT_FALSE(buffer3.has_errors());
  EXPECT_THAT(buffer3, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Less},
                           {.kind = TokenKind::LessEqualGreater},
                           {.kind = TokenKind::Greater},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer4 = compile_helper_.GetTokenizedBuffer("\\/?@&^!");
  EXPECT_FALSE(buffer4.has_errors());
  EXPECT_THAT(buffer4, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Backslash},
                           {.kind = TokenKind::Slash},
                           {.kind = TokenKind::Question},
                           {.kind = TokenKind::At},
                           {.kind = TokenKind::Amp},
                           {.kind = TokenKind::Caret},
                           {.kind = TokenKind::Exclaim},
                           {.kind = TokenKind::FileEnd},
                       }));
}

TEST_F(LexerTest, Parens) {
  auto& buffer1 = compile_helper_.GetTokenizedBuffer("()");
  EXPECT_FALSE(buffer1.has_errors());
  EXPECT_THAT(buffer1, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::OpenParen},
                           {.kind = TokenKind::CloseParen},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer2 = compile_helper_.GetTokenizedBuffer("((()()))");
  EXPECT_FALSE(buffer2.has_errors());
  EXPECT_THAT(buffer2, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::OpenParen},
                           {.kind = TokenKind::OpenParen},
                           {.kind = TokenKind::OpenParen},
                           {.kind = TokenKind::CloseParen},
                           {.kind = TokenKind::OpenParen},
                           {.kind = TokenKind::CloseParen},
                           {.kind = TokenKind::CloseParen},
                           {.kind = TokenKind::CloseParen},
                           {.kind = TokenKind::FileEnd},
                       }));
}

TEST_F(LexerTest, CurlyBraces) {
  auto& buffer1 = compile_helper_.GetTokenizedBuffer("{}");
  EXPECT_FALSE(buffer1.has_errors());
  EXPECT_THAT(buffer1, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::OpenCurlyBrace},
                           {.kind = TokenKind::CloseCurlyBrace},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer2 = compile_helper_.GetTokenizedBuffer("{{{}{}}}");
  EXPECT_FALSE(buffer2.has_errors());
  EXPECT_THAT(buffer2, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::OpenCurlyBrace},
                           {.kind = TokenKind::OpenCurlyBrace},
                           {.kind = TokenKind::OpenCurlyBrace},
                           {.kind = TokenKind::CloseCurlyBrace},
                           {.kind = TokenKind::OpenCurlyBrace},
                           {.kind = TokenKind::CloseCurlyBrace},
                           {.kind = TokenKind::CloseCurlyBrace},
                           {.kind = TokenKind::CloseCurlyBrace},
                           {.kind = TokenKind::FileEnd},
                       }));
}

TEST_F(LexerTest, MatchingGroups) {
  {
    auto& buffer = compile_helper_.GetTokenizedBuffer("(){}");
    ASSERT_FALSE(buffer.has_errors());
    auto it = ++buffer.tokens().begin();
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
    auto eof_token = *it++;
    EXPECT_EQ(buffer.GetKind(eof_token), TokenKind::FileEnd);
    EXPECT_EQ(buffer.tokens().end(), it);
  }

  {
    auto [buffer, value_stores] =
        compile_helper_.GetTokenizedBufferWithSharedValueStore(
            "({x}){(y)} {{((z))}}");
    ASSERT_FALSE(buffer.has_errors());
    auto it = ++buffer.tokens().begin();
    auto open_paren_token = *it++;
    auto open_curly_token = *it++;

    ASSERT_EQ("x", value_stores.identifiers().Get(buffer.GetIdentifier(*it++)));
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
    ASSERT_EQ("y", value_stores.identifiers().Get(buffer.GetIdentifier(*it++)));
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
    ASSERT_EQ("z", value_stores.identifiers().Get(buffer.GetIdentifier(*it++)));
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

    auto eof_token = *it++;
    EXPECT_EQ(buffer.GetKind(eof_token), TokenKind::FileEnd);
    EXPECT_EQ(buffer.tokens().end(), it);
  }
}

TEST_F(LexerTest, MismatchedGroups) {
  auto& buffer1 = compile_helper_.GetTokenizedBuffer("{");
  EXPECT_TRUE(buffer1.has_errors());
  EXPECT_THAT(buffer1, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Error, .text = "{"},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer2 = compile_helper_.GetTokenizedBuffer("}");
  EXPECT_TRUE(buffer2.has_errors());
  EXPECT_THAT(buffer2, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Error, .text = "}"},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer3 = compile_helper_.GetTokenizedBuffer("{(}");
  EXPECT_TRUE(buffer3.has_errors());
  EXPECT_THAT(
      buffer3,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart},
          {.kind = TokenKind::OpenCurlyBrace, .column = 1},
          {.kind = TokenKind::OpenParen, .column = 2},
          {.kind = TokenKind::CloseParen, .column = 3, .recovery = true},
          {.kind = TokenKind::CloseCurlyBrace, .column = 3},
          {.kind = TokenKind::FileEnd},
      }));

  auto& buffer4 = compile_helper_.GetTokenizedBuffer(")({)");
  EXPECT_TRUE(buffer4.has_errors());
  EXPECT_THAT(
      buffer4,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart},
          {.kind = TokenKind::Error, .column = 1, .text = ")"},
          {.kind = TokenKind::OpenParen, .column = 2},
          {.kind = TokenKind::OpenCurlyBrace, .column = 3},
          {.kind = TokenKind::CloseCurlyBrace, .column = 4, .recovery = true},
          {.kind = TokenKind::CloseParen, .column = 4},
          {.kind = TokenKind::FileEnd},
      }));
}

TEST_F(LexerTest, Whitespace) {
  auto& buffer = compile_helper_.GetTokenizedBuffer("{( } {(");

  // Whether there should be whitespace before/after each token.
  bool space[] = {false,
                  // start-of-file
                  true,
                  // {
                  false,
                  // (
                  true,
                  // inserted )
                  true,
                  // }
                  true,
                  // error {
                  false,
                  // error (
                  true,
                  // EOF
                  false};
  int pos = 0;
  for (TokenIndex token : buffer.tokens()) {
    SCOPED_TRACE(
        llvm::formatv("Token #{0}: '{1}'", token, buffer.GetTokenText(token)));

    ASSERT_LT(pos, std::size(space));
    EXPECT_THAT(buffer.HasLeadingWhitespace(token), Eq(space[pos]));
    ++pos;
    ASSERT_LT(pos, std::size(space));
    EXPECT_THAT(buffer.HasTrailingWhitespace(token), Eq(space[pos]));
  }
  ASSERT_EQ(pos + 1, std::size(space));
}

TEST_F(LexerTest, Keywords) {
  TokenKind keywords[] = {
#define CARBON_TOKEN(TokenName)
#define CARBON_KEYWORD_TOKEN(TokenName, ...) TokenKind::TokenName,
#include "toolchain/lex/token_kind.def"
  };
  for (const auto& keyword : keywords) {
    auto& buffer = compile_helper_.GetTokenizedBuffer(keyword.fixed_spelling());
    EXPECT_FALSE(buffer.has_errors());
    EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::FileStart},
                            {.kind = keyword, .column = 1, .indent_column = 1},
                            {.kind = TokenKind::FileEnd},
                        }));
  }
}

TEST_F(LexerTest, Comments) {
  auto& buffer1 = compile_helper_.GetTokenizedBuffer(" ;\n  // foo\n  ;\n");
  EXPECT_FALSE(buffer1.has_errors());
  EXPECT_THAT(
      buffer1,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart, .line = 1, .column = 1},
          {.kind = TokenKind::Semi, .line = 1, .column = 2, .indent_column = 2},
          {.kind = TokenKind::Semi, .line = 3, .column = 3, .indent_column = 3},
          {.kind = TokenKind::FileEnd, .line = 3, .column = 4},
      }));

  auto& buffer2 = compile_helper_.GetTokenizedBuffer("// foo\n//\n// bar");
  EXPECT_FALSE(buffer2.has_errors());
  EXPECT_THAT(buffer2, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::FileEnd}}));

  // Make sure weird characters aren't a problem.
  auto& buffer3 =
      compile_helper_.GetTokenizedBuffer("  // foo#$!^?@-_💩🍫⃠ [̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅]");
  EXPECT_FALSE(buffer3.has_errors());
  EXPECT_THAT(buffer3, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::FileEnd}}));

  // Make sure we can lex a comment at the end of the input.
  auto& buffer4 = compile_helper_.GetTokenizedBuffer("//");
  EXPECT_FALSE(buffer4.has_errors());
  EXPECT_THAT(buffer4, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::FileEnd}}));
}

TEST_F(LexerTest, InvalidComments) {
  llvm::StringLiteral testcases[] = {
      "  /// foo\n",
      "foo // bar\n",
      "//! hello",
      " //world",
  };
  for (llvm::StringLiteral testcase : testcases) {
    auto& buffer = compile_helper_.GetTokenizedBuffer(testcase);
    EXPECT_TRUE(buffer.has_errors());
  }
}

TEST_F(LexerTest, Identifiers) {
  auto& buffer1 = compile_helper_.GetTokenizedBuffer("   foobar");
  EXPECT_FALSE(buffer1.has_errors());
  EXPECT_THAT(buffer1, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Identifier,
                            .column = 4,
                            .indent_column = 4,
                            .text = "foobar"},
                           {.kind = TokenKind::FileEnd},
                       }));

  // Check different kinds of identifier character sequences.
  auto& buffer2 = compile_helper_.GetTokenizedBuffer("_foo_bar");
  EXPECT_FALSE(buffer2.has_errors());
  EXPECT_THAT(buffer2, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Identifier, .text = "_foo_bar"},
                           {.kind = TokenKind::FileEnd},
                       }));

  auto& buffer3 = compile_helper_.GetTokenizedBuffer("foo2bar00");
  EXPECT_FALSE(buffer3.has_errors());
  EXPECT_THAT(buffer3, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Identifier, .text = "foo2bar00"},
                           {.kind = TokenKind::FileEnd},
                       }));

  // Check that we can parse identifiers that start with a keyword.
  auto& buffer4 = compile_helper_.GetTokenizedBuffer("fnord");
  EXPECT_FALSE(buffer4.has_errors());
  EXPECT_THAT(buffer4, HasTokens(llvm::ArrayRef<ExpectedToken>{
                           {.kind = TokenKind::FileStart},
                           {.kind = TokenKind::Identifier, .text = "fnord"},
                           {.kind = TokenKind::FileEnd},
                       }));

  // Check multiple identifiers with indent and interning.
  auto& buffer5 =
      compile_helper_.GetTokenizedBuffer("   foo;bar\nbar \n  foo\tfoo");
  EXPECT_FALSE(buffer5.has_errors());
  EXPECT_THAT(buffer5,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},
                  {.kind = TokenKind::Identifier,
                   .line = 1,
                   .column = 4,
                   .indent_column = 4,
                   .text = "foo"},
                  {.kind = TokenKind::Semi},
                  {.kind = TokenKind::Identifier,
                   .line = 1,
                   .column = 8,
                   .indent_column = 4,
                   .text = "bar"},
                  {.kind = TokenKind::Identifier,
                   .line = 2,
                   .column = 1,
                   .indent_column = 1,
                   .text = "bar"},
                  {.kind = TokenKind::Identifier,
                   .line = 3,
                   .column = 3,
                   .indent_column = 3,
                   .text = "foo"},
                  {.kind = TokenKind::Identifier,
                   .line = 3,
                   .column = 7,
                   .indent_column = 3,
                   .text = "foo"},
                  {.kind = TokenKind::FileEnd, .line = 3, .column = 10},
              }));
}

TEST_F(LexerTest, StringLiterals) {
  llvm::StringLiteral testcase = R"(
    "hello world\n"

    '''foo
      test \
      \xAB
     ''' trailing

      #"""#

    "\0"

    #"\0"foo"\1"#

    """x"""
  )";

  auto [buffer, value_stores] =
      compile_helper_.GetTokenizedBufferWithSharedValueStore(testcase);
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},
                  {.kind = TokenKind::StringLiteral,
                   .line = 2,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = {"hello world\n"}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 4,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = {" test  \xAB\n"}},
                  {.kind = TokenKind::Identifier,
                   .line = 7,
                   .column = 10,
                   .indent_column = 5,
                   .text = "trailing"},
                  {.kind = TokenKind::StringLiteral,
                   .line = 9,
                   .column = 7,
                   .indent_column = 7,
                   .value_stores = &value_stores,
                   .string_contents = {"\""}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 11,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = llvm::StringLiteral::withInnerNUL("\0")},
                  {.kind = TokenKind::StringLiteral,
                   .line = 13,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = {"\\0\"foo\"\\1"}},

                  // """x""" is three string literals, not one invalid
                  // attempt at a block string literal.
                  {.kind = TokenKind::StringLiteral,
                   .line = 15,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = {""}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 15,
                   .column = 7,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = {"x"}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 15,
                   .column = 10,
                   .indent_column = 5,
                   .value_stores = &value_stores,
                   .string_contents = {""}},
                  {.kind = TokenKind::FileEnd, .line = 16, .column = 3},
              }));
}

TEST_F(LexerTest, InvalidStringLiterals) {
  llvm::StringLiteral invalid[] = {
      // clang-format off
      R"(")",
      R"('''
      '')",
      R"("\)",
      R"("\")",
      R"("\\)",
      R"("\\\")",
      R"(''')",
      R"('''
      )",
      R"('''\)",
      R"(#'''
      ''')",
      // clang-format on
  };

  for (llvm::StringLiteral test : invalid) {
    SCOPED_TRACE(test);
    auto& buffer = compile_helper_.GetTokenizedBuffer(test);
    EXPECT_TRUE(buffer.has_errors());

    // We should have formed at least one error token.
    bool found_error = false;
    for (TokenIndex token : buffer.tokens()) {
      if (buffer.GetKind(token) == TokenKind::Error) {
        found_error = true;
        break;
      }
    }
    EXPECT_TRUE(found_error);
  }
}

TEST_F(LexerTest, TypeLiterals) {
  llvm::StringLiteral testcase = R"(
    i0 i1 i20 i999999999999 i0x1
    u0 u1 u64 u64b
    f32 f80 f1 fi
    s1
  )";

  auto [buffer, value_stores] =
      compile_helper_.GetTokenizedBufferWithSharedValueStore(testcase);
  EXPECT_FALSE(buffer.has_errors());
  ASSERT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},

                  {.kind = TokenKind::Identifier,
                   .line = 2,
                   .column = 5,
                   .indent_column = 5,
                   .text = {"i0"}},
                  {.kind = TokenKind::IntTypeLiteral,
                   .line = 2,
                   .column = 8,
                   .indent_column = 5,
                   .text = {"i1"}},
                  {.kind = TokenKind::IntTypeLiteral,
                   .line = 2,
                   .column = 11,
                   .indent_column = 5,
                   .text = {"i20"}},
                  {.kind = TokenKind::IntTypeLiteral,
                   .line = 2,
                   .column = 15,
                   .indent_column = 5,
                   .text = {"i999999999999"}},
                  {.kind = TokenKind::Identifier,
                   .line = 2,
                   .column = 29,
                   .indent_column = 5,
                   .text = {"i0x1"}},

                  {.kind = TokenKind::Identifier,
                   .line = 3,
                   .column = 5,
                   .indent_column = 5,
                   .text = {"u0"}},
                  {.kind = TokenKind::UnsignedIntTypeLiteral,
                   .line = 3,
                   .column = 8,
                   .indent_column = 5,
                   .text = {"u1"}},
                  {.kind = TokenKind::UnsignedIntTypeLiteral,
                   .line = 3,
                   .column = 11,
                   .indent_column = 5,
                   .text = {"u64"}},
                  {.kind = TokenKind::Identifier,
                   .line = 3,
                   .column = 15,
                   .indent_column = 5,
                   .text = {"u64b"}},

                  {.kind = TokenKind::FloatTypeLiteral,
                   .line = 4,
                   .column = 5,
                   .indent_column = 5,
                   .text = {"f32"}},
                  {.kind = TokenKind::FloatTypeLiteral,
                   .line = 4,
                   .column = 9,
                   .indent_column = 5,
                   .text = {"f80"}},
                  {.kind = TokenKind::FloatTypeLiteral,
                   .line = 4,
                   .column = 13,
                   .indent_column = 5,
                   .text = {"f1"}},
                  {.kind = TokenKind::Identifier,
                   .line = 4,
                   .column = 16,
                   .indent_column = 5,
                   .text = {"fi"}},

                  {.kind = TokenKind::Identifier,
                   .line = 5,
                   .column = 5,
                   .indent_column = 5,
                   .text = {"s1"}},

                  {.kind = TokenKind::FileEnd, .line = 6, .column = 3},
              }));

  auto type_size = [&](int token_index) {
    auto token = buffer.tokens().begin()[token_index];
    return value_stores.ints().Get(buffer.GetTypeLiteralSize(token));
  };

  EXPECT_EQ(type_size(2), 1);
  EXPECT_EQ(type_size(3), 20);
  EXPECT_EQ(type_size(4), 999999999999ULL);
  EXPECT_EQ(type_size(7), 1);
  EXPECT_EQ(type_size(8), 64);
  EXPECT_EQ(type_size(10), 32);
  EXPECT_EQ(type_size(11), 80);
  EXPECT_EQ(type_size(12), 1);
}

TEST_F(LexerTest, TypeLiteralTooManyDigits) {
  // We increase the number of digits until the first one that is to large.
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::TooManyTypeBitWidthDigits,
                            Diagnostics::Level::Error, 1, 2, _)));
  std::string code = "i";
  // A 128-bit APInt should be plenty large, but if needed in the future it can
  // be widened without issue.
  llvm::APInt bits = llvm::APInt::getZero(128);
  for ([[maybe_unused]] auto _ : llvm::seq(1, 30)) {
    code.append("9");
    bits = bits * 10 + 9;
    auto [buffer, value_stores] =
        compile_helper_.GetTokenizedBufferWithSharedValueStore(code, &consumer);
    if (buffer.has_errors()) {
      ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                              {.kind = TokenKind::FileStart},
                              {.kind = TokenKind::Error, .text = code},
                              {.kind = TokenKind::FileEnd},
                          }));
      break;
    }
    ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                            {.kind = TokenKind::FileStart},
                            {.kind = TokenKind::IntTypeLiteral, .text = code},
                            {.kind = TokenKind::FileEnd},
                        }));
    auto token = buffer.tokens().begin()[1];
    EXPECT_TRUE(llvm::APInt::isSameValue(
        value_stores.ints().Get(buffer.GetTypeLiteralSize(token)), bits));
  }

  // Make sure we can also gracefully reject very large number of digits without
  // crashing or hanging, and show the correct number.
  constexpr int Count = 10000;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::TooManyTypeBitWidthDigits,
                            Diagnostics::Level::Error, 1, 2,
                            HasSubstr(llvm::formatv(" {0} ", Count)))));
  code = "i";
  code.append(Count, '9');
  auto& buffer = compile_helper_.GetTokenizedBuffer(code, &consumer);
  ASSERT_TRUE(buffer.has_errors());
  ASSERT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::FileStart},
                          {.kind = TokenKind::Error, .text = code},
                          {.kind = TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, DiagnosticTrailingComment) {
  llvm::StringLiteral testcase = R"(
    // Hello!
    var String x; // trailing comment
  )";

  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::TrailingComment,
                            Diagnostics::Level::Error, 3, 19, _)));
  compile_helper_.GetTokenizedBuffer(testcase, &consumer);
}

TEST_F(LexerTest, DiagnosticWhitespace) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer,
              HandleDiagnostic(IsSingleDiagnostic(
                  Diagnostics::Kind::NoWhitespaceAfterCommentIntroducer,
                  Diagnostics::Level::Error, 1, 3, _)));
  compile_helper_.GetTokenizedBuffer("//no space after comment", &consumer);
}

TEST_F(LexerTest, DiagnosticUnrecognizedEscape) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer,
              HandleDiagnostic(IsSingleDiagnostic(
                  Diagnostics::Kind::UnknownEscapeSequence,
                  Diagnostics::Level::Error, 1, 8, HasSubstr("`b`"))));
  compile_helper_.GetTokenizedBuffer(R"("hello\bworld")", &consumer);
}

TEST_F(LexerTest, DiagnosticBadHex) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::HexadecimalEscapeMissingDigits,
                            Diagnostics::Level::Error, 1, 9, _)));
  compile_helper_.GetTokenizedBuffer(R"("hello\xabworld")", &consumer);
}

TEST_F(LexerTest, DiagnosticInvalidDigit) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer,
              HandleDiagnostic(IsSingleDiagnostic(
                  Diagnostics::Kind::InvalidDigit, Diagnostics::Level::Error, 1,
                  6, HasSubstr("'a'"))));
  compile_helper_.GetTokenizedBuffer("0x123abc", &consumer);
}

TEST_F(LexerTest, DiagnosticCR) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::UnsupportedCrLineEnding,
                            Diagnostics::Level::Error, 1, 1, _)));
  compile_helper_.GetTokenizedBuffer("\r", &consumer);
}

TEST_F(LexerTest, DiagnosticLfCr) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::UnsupportedLfCrLineEnding,
                            Diagnostics::Level::Error, 2, 1, _)));
  compile_helper_.GetTokenizedBuffer("\n\r", &consumer);
}

TEST_F(LexerTest, DiagnosticMissingTerminator) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::UnterminatedString,
                            Diagnostics::Level::Error, 1, 1, _)));
  compile_helper_.GetTokenizedBuffer(R"(#" ")", &consumer);
}

TEST_F(LexerTest, DiagnosticUnrecognizedChar) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Diagnostics::Kind::UnrecognizedCharacters,
                            Diagnostics::Level::Error, 1, 1, _)));
  compile_helper_.GetTokenizedBuffer("\b", &consumer);
}

TEST_F(LexerTest, DiagnosticFileTooLarge) {
  Testing::MockDiagnosticConsumer consumer;
  static constexpr size_t NumLines = 10'000'000;
  std::string input;
  input.reserve(NumLines * 3);
  for ([[maybe_unused]] auto _ : llvm::seq(NumLines)) {
    input += "{}\n";
  }
  EXPECT_CALL(consumer,
              HandleDiagnostic(IsSingleDiagnostic(
                  Diagnostics::Kind::TooManyTokens, Diagnostics::Level::Error,
                  TokenIndex::Max / 2, 1, _)));
  compile_helper_.GetTokenizedBuffer(input, &consumer);
}

// Outputs comments to the stream to create a comment block.
static auto AppendCommentLines(RawStringOstream& out, int count,
                               llvm::StringRef tag) -> void {
  for (int i : llvm::seq(count)) {
    out << "// " << tag << i << "\n";
  }
}

TEST_F(LexerTest, CommentBlock) {
  for (int comments_before = 0; comments_before < 5; ++comments_before) {
    RawStringOstream prefix;
    AppendCommentLines(prefix, comments_before, "B");
    std::string prefix_out = prefix.TakeStr();

    for (int comments_after = 1; comments_after < 5; ++comments_after) {
      RawStringOstream source;
      source << prefix_out;
      if (comments_before > 0) {
        source << "//\n";
      }
      AppendCommentLines(source, comments_after, "C");

      SCOPED_TRACE(llvm::formatv(
          "{0} comment lines before the empty comment line, {1} after",
          comments_before, comments_after));

      auto& buffer = compile_helper_.GetTokenizedBuffer(source.TakeStr());
      ASSERT_FALSE(buffer.has_errors());

      EXPECT_THAT(buffer.comments_size(), Eq(1));
    }
  }
}

TEST_F(LexerTest, IndentedComments) {
  for (int indent = 0; indent < 40; ++indent) {
    SCOPED_TRACE(llvm::formatv("Indent: {0}", indent));

    RawStringOstream source;
    source.indent(indent);
    source << "// Comment\n";
    std::string source_str = source.TakeStr();

    auto& buffer = compile_helper_.GetTokenizedBuffer(source_str);
    ASSERT_FALSE(buffer.has_errors());
    EXPECT_THAT(buffer.comments_size(), Eq(1));

    std::string simd_source =
        source_str +
        "\"Add a bunch of padding so that SIMD logic shouldn't hit EOF\"";
    auto& simd_buffer = compile_helper_.GetTokenizedBuffer(simd_source);
    ASSERT_FALSE(simd_buffer.has_errors());
    EXPECT_THAT(simd_buffer.comments_size(), Eq(1));
  }
}

TEST_F(LexerTest, MultipleComments) {
  // TODO: Switch format to `llvm::StringLiteral` if
  // `llvm::StringLiteral::c_str` is added.
  constexpr char Format[] = R"(
{0}
  {1}

{2}
                                                              {3}

'''This is a string, not a comment. The next comment will stop SIMD due to being
   too close to the EOF.
   '''

{4}
x
)";
  constexpr llvm::StringLiteral Comments[] = {
      // NOLINTNEXTLINE(bugprone-suspicious-missing-comma)
      "// This comment should be possible to parse with SIMD.\n"
      "// This one too.\n",
      "// This one as well, though it's a different indent.\n"
      "        // And mixes indent.\n"
      "   // And mixes indent more.\n",
      "// This is one comment:\n"
      "//Invalid\n"
      "// Valid\n"
      "//Invalid\n"
      "//\n"
      "// Valid\n"
      "//\n"
      "// Valid\n",
      "// This uses a high indent, which stops SIMD.\n", "//\n"};
  std::string source = llvm::formatv(Format, Comments[0], Comments[1],
                                     Comments[2], Comments[3], Comments[4])
                           .str();

  auto& buffer = compile_helper_.GetTokenizedBuffer(source);
  EXPECT_TRUE(buffer.has_errors());

  EXPECT_THAT(buffer.comments_size(), Eq(std::size(Comments)));
  for (int i :
       llvm::seq(std::min<int>(buffer.comments_size(), std::size(Comments)))) {
    EXPECT_THAT(buffer.GetCommentText(CommentIndex(i)).str(),
                testing::StrEq(Comments[i]));
  }
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {.kind = TokenKind::FileStart},
                          {.kind = TokenKind::StringLiteral},
                          {.kind = TokenKind::Identifier},
                          {.kind = TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, PrintingOutputYaml) {
  // Test that we can parse this into YAML and verify line and indent data.
  auto& buffer =
      compile_helper_.GetTokenizedBuffer("\n ;\n\n\n; ;\n\n\n\n\n\n\n\n\n\n\n");
  ASSERT_FALSE(buffer.has_errors());
  RawStringOstream print_stream;
  buffer.Print(print_stream);

  EXPECT_THAT(
      Yaml::Value::FromText(print_stream.TakeStr()),
      IsYaml(ElementsAre(Yaml::Sequence(ElementsAre(Yaml::Mapping(ElementsAre(
          Pair("filename", buffer.source().filename().str()),
          Pair("tokens", Yaml::Sequence(ElementsAre(
                             Yaml::Mapping(ElementsAre(
                                 Pair("index", "0"), Pair("kind", "FileStart"),
                                 Pair("line", "1"), Pair("column", "1"),
                                 Pair("indent", "1"), Pair("spelling", ""))),
                             Yaml::Mapping(ElementsAre(
                                 Pair("index", "1"), Pair("kind", "Semi"),
                                 Pair("line", "2"), Pair("column", "2"),
                                 Pair("indent", "2"), Pair("spelling", ";"),
                                 Pair("has_leading_space", "true"))),
                             Yaml::Mapping(ElementsAre(
                                 Pair("index", "2"), Pair("kind", "Semi"),
                                 Pair("line", "5"), Pair("column", "1"),
                                 Pair("indent", "1"), Pair("spelling", ";"),
                                 Pair("has_leading_space", "true"))),
                             Yaml::Mapping(ElementsAre(
                                 Pair("index", "3"), Pair("kind", "Semi"),
                                 Pair("line", "5"), Pair("column", "3"),
                                 Pair("indent", "1"), Pair("spelling", ";"),
                                 Pair("has_leading_space", "true"))),
                             Yaml::Mapping(ElementsAre(
                                 Pair("index", "4"), Pair("kind", "FileEnd"),
                                 Pair("line", "15"), Pair("column", "1"),
                                 Pair("indent", "1"), Pair("spelling", ""),
                                 Pair("has_leading_space", "true")))))))))))));
}

}  // namespace
}  // namespace Carbon::Lex
