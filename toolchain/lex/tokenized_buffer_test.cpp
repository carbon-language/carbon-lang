// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>
#include <iterator>

#include "llvm/ADT/ArrayRef.h"
#include "testing/base/test_raw_ostream.h"
#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer_test_helpers.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::Lex {
namespace {

using ::Carbon::Testing::ExpectedToken;
using ::Carbon::Testing::IsSingleDiagnostic;
using ::Carbon::Testing::TestRawOstream;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Pair;

namespace Yaml = ::Carbon::Testing::Yaml;

class LexerTest : public ::testing::Test {
 protected:
  auto GetSourceBuffer(llvm::StringRef text) -> SourceBuffer& {
    std::string filename = llvm::formatv("test{0}.carbon", ++file_index_);
    CARBON_CHECK(fs_.addFile(filename, /*ModificationTime=*/0,
                             llvm::MemoryBuffer::getMemBuffer(text)));
    source_storage_.push_front(std::move(*SourceBuffer::MakeFromFile(
        fs_, filename, ConsoleDiagnosticConsumer())));
    return source_storage_.front();
  }

  auto Lex(llvm::StringRef text,
           DiagnosticConsumer& consumer = ConsoleDiagnosticConsumer())
      -> TokenizedBuffer {
    return Lex::Lex(value_stores_, GetSourceBuffer(text), consumer);
  }

  SharedValueStores value_stores_;
  llvm::vfs::InMemoryFileSystem fs_;
  int file_index_ = 0;
  std::forward_list<SourceBuffer> source_storage_;
};

TEST_F(LexerTest, HandlesEmptyBuffer) {
  auto buffer = Lex("");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart}, {TokenKind::FileEnd}}));
}

TEST_F(LexerTest, TracksLinesAndColumns) {
  auto buffer = Lex("\n  ;;\n   ;;;\n   x\"foo\" '''baz\n  a\n ''' y");
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

TEST_F(LexerTest, HandlesNumericLiteral) {
  auto buffer = Lex("12-578\n  1  2\n0x12_3ABC\n0b10_10_11\n1_234_567\n1.5e9");
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
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_12)), 12);
  auto token_578 = token_12 + 2;
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_578)), 578);
  auto token_1 = token_578 + 1;
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_1)), 1);
  auto token_2 = token_1 + 1;
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_2)), 2);
  auto token_0x12_3abc = token_2 + 1;
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_0x12_3abc)),
            0x12'3abc);
  auto token_0b10_10_11 = token_0x12_3abc + 1;
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_0b10_10_11)),
            0b10'10'11);
  auto token_1_234_567 = token_0b10_10_11 + 1;
  EXPECT_EQ(value_stores_.ints().Get(buffer.GetIntLiteral(*token_1_234_567)),
            1'234'567);
  auto token_1_5e9 = token_1_234_567 + 1;
  auto value_1_5e9 =
      value_stores_.reals().Get(buffer.GetRealLiteral(*token_1_5e9));
  EXPECT_EQ(value_1_5e9.mantissa.getZExtValue(), 15);
  EXPECT_EQ(value_1_5e9.exponent.getSExtValue(), 8);
  EXPECT_EQ(value_1_5e9.is_decimal, true);
}

TEST_F(LexerTest, HandlesInvalidNumericLiterals) {
  auto buffer = Lex("14x 15_49 0x3.5q 0x3_4.5_6 0ops");
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
  auto buffer = Lex(source_text);
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
                          {.kind = TokenKind::RealLiteral, .text = "9.5"},
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
  auto buffer = Lex(llvm::StringRef(GarbageText, sizeof(GarbageText) - 1));
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
  auto buffer = Lex("<<<");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::LessLess},
                          {TokenKind::Less},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("<<=>>");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::LessLessEqual},
                          {TokenKind::GreaterGreater},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("< <=> >");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::Less},
                          {TokenKind::LessEqualGreater},
                          {TokenKind::Greater},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("\\/?@&^!");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::Backslash},
                          {TokenKind::Slash},
                          {TokenKind::Question},
                          {TokenKind::At},
                          {TokenKind::Amp},
                          {TokenKind::Caret},
                          {TokenKind::Exclaim},
                          {TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, Parens) {
  auto buffer = Lex("()");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::OpenParen},
                          {TokenKind::CloseParen},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("((()()))");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::OpenParen},
                          {TokenKind::OpenParen},
                          {TokenKind::OpenParen},
                          {TokenKind::CloseParen},
                          {TokenKind::OpenParen},
                          {TokenKind::CloseParen},
                          {TokenKind::CloseParen},
                          {TokenKind::CloseParen},
                          {TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, CurlyBraces) {
  auto buffer = Lex("{}");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::OpenCurlyBrace},
                          {TokenKind::CloseCurlyBrace},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("{{{}{}}}");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::OpenCurlyBrace},
                          {TokenKind::OpenCurlyBrace},
                          {TokenKind::OpenCurlyBrace},
                          {TokenKind::CloseCurlyBrace},
                          {TokenKind::OpenCurlyBrace},
                          {TokenKind::CloseCurlyBrace},
                          {TokenKind::CloseCurlyBrace},
                          {TokenKind::CloseCurlyBrace},
                          {TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, MatchingGroups) {
  {
    TokenizedBuffer buffer = Lex("(){}");
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
    TokenizedBuffer buffer = Lex("({x}){(y)} {{((z))}}");
    ASSERT_FALSE(buffer.has_errors());
    auto it = ++buffer.tokens().begin();
    auto open_paren_token = *it++;
    auto open_curly_token = *it++;

    ASSERT_EQ("x",
              value_stores_.identifiers().Get(buffer.GetIdentifier(*it++)));
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
    ASSERT_EQ("y",
              value_stores_.identifiers().Get(buffer.GetIdentifier(*it++)));
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
    ASSERT_EQ("z",
              value_stores_.identifiers().Get(buffer.GetIdentifier(*it++)));
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
  auto buffer = Lex("{");
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {.kind = TokenKind::Error, .text = "{"},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("}");
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {.kind = TokenKind::Error, .text = "}"},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("{(}");
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {TokenKind::FileStart},
          {.kind = TokenKind::OpenCurlyBrace, .column = 1},
          {.kind = TokenKind::OpenParen, .column = 2},
          {.kind = TokenKind::CloseParen, .column = 3, .recovery = true},
          {.kind = TokenKind::CloseCurlyBrace, .column = 3},
          {TokenKind::FileEnd},
      }));

  buffer = Lex(")({)");
  EXPECT_TRUE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {TokenKind::FileStart},
          {.kind = TokenKind::Error, .column = 1, .text = ")"},
          {.kind = TokenKind::OpenParen, .column = 2},
          {.kind = TokenKind::OpenCurlyBrace, .column = 3},
          {.kind = TokenKind::CloseCurlyBrace, .column = 4, .recovery = true},
          {.kind = TokenKind::CloseParen, .column = 4},
          {TokenKind::FileEnd},
      }));
}

TEST_F(LexerTest, Whitespace) {
  auto buffer = Lex("{( } {(");

  // Whether there should be whitespace before/after each token.
  bool space[] = {true,
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
  auto buffer = Lex("   fn");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {TokenKind::FileStart},
                  {.kind = TokenKind::Fn, .column = 4, .indent_column = 4},
                  {TokenKind::FileEnd},
              }));

  buffer = Lex("and or not if else for return var break continue _");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {TokenKind::And},
                          {TokenKind::Or},
                          {TokenKind::Not},
                          {TokenKind::If},
                          {TokenKind::Else},
                          {TokenKind::For},
                          {TokenKind::Return},
                          {TokenKind::Var},
                          {TokenKind::Break},
                          {TokenKind::Continue},
                          {TokenKind::Underscore},
                          {TokenKind::FileEnd},
                      }));
}

TEST_F(LexerTest, Comments) {
  auto buffer = Lex(" ;\n  // foo\n  ;\n");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(
      buffer,
      HasTokens(llvm::ArrayRef<ExpectedToken>{
          {.kind = TokenKind::FileStart, .line = 1, .column = 1},
          {.kind = TokenKind::Semi, .line = 1, .column = 2, .indent_column = 2},
          {.kind = TokenKind::Semi, .line = 3, .column = 3, .indent_column = 3},
          {.kind = TokenKind::FileEnd, .line = 3, .column = 4},
      }));

  buffer = Lex("// foo\n//\n// bar");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart}, {TokenKind::FileEnd}}));

  // Make sure weird characters aren't a problem.
  buffer = Lex("  // foo#$!^?@-_💩🍫⃠ [̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅]");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart}, {TokenKind::FileEnd}}));

  // Make sure we can lex a comment at the end of the input.
  buffer = Lex("//");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart}, {TokenKind::FileEnd}}));
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
    EXPECT_TRUE(buffer.has_errors());
  }
}

TEST_F(LexerTest, Identifiers) {
  auto buffer = Lex("   foobar");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {.kind = TokenKind::Identifier,
                           .column = 4,
                           .indent_column = 4,
                           .text = "foobar"},
                          {TokenKind::FileEnd},
                      }));

  // Check different kinds of identifier character sequences.
  buffer = Lex("_foo_bar");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {.kind = TokenKind::Identifier, .text = "_foo_bar"},
                          {TokenKind::FileEnd},
                      }));

  buffer = Lex("foo2bar00");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {.kind = TokenKind::Identifier, .text = "foo2bar00"},
                          {TokenKind::FileEnd},
                      }));

  // Check that we can parse identifiers that start with a keyword.
  buffer = Lex("fnord");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer, HasTokens(llvm::ArrayRef<ExpectedToken>{
                          {TokenKind::FileStart},
                          {.kind = TokenKind::Identifier, .text = "fnord"},
                          {TokenKind::FileEnd},
                      }));

  // Check multiple identifiers with indent and interning.
  buffer = Lex("   foo;bar\nbar \n  foo\tfoo");
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer,
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

  auto buffer = Lex(testcase);
  EXPECT_FALSE(buffer.has_errors());
  EXPECT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},
                  {.kind = TokenKind::StringLiteral,
                   .line = 2,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
                   .string_contents = {"hello world\n"}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 4,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
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
                   .value_stores = &value_stores_,
                   .string_contents = {"\""}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 11,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
                   .string_contents = llvm::StringLiteral::withInnerNUL("\0")},
                  {.kind = TokenKind::StringLiteral,
                   .line = 13,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
                   .string_contents = {"\\0\"foo\"\\1"}},

                  // """x""" is three string literals, not one invalid
                  // attempt at a block string literal.
                  {.kind = TokenKind::StringLiteral,
                   .line = 15,
                   .column = 5,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
                   .string_contents = {""}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 15,
                   .column = 7,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
                   .string_contents = {"x"}},
                  {.kind = TokenKind::StringLiteral,
                   .line = 15,
                   .column = 10,
                   .indent_column = 5,
                   .value_stores = &value_stores_,
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
    auto buffer = Lex(test);
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

  auto buffer = Lex(testcase);
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
    return value_stores_.ints().Get(buffer.GetTypeLiteralSize(token));
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
  std::string code = "i";
  constexpr int Count = 10000;
  code.append(Count, '9');

  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer,
              HandleDiagnostic(IsSingleDiagnostic(
                  DiagnosticKind::TooManyDigits, DiagnosticLevel::Error, 1, 2,
                  HasSubstr(llvm::formatv(" {0} ", Count)))));
  auto buffer = Lex(code, consumer);
  EXPECT_TRUE(buffer.has_errors());
  ASSERT_THAT(buffer,
              HasTokens(llvm::ArrayRef<ExpectedToken>{
                  {.kind = TokenKind::FileStart, .line = 1, .column = 1},
                  {.kind = TokenKind::Error,
                   .line = 1,
                   .column = 1,
                   .indent_column = 1,
                   .text = {code}},
                  {.kind = TokenKind::FileEnd, .line = 1, .column = Count + 2},
              }));
}

TEST_F(LexerTest, DiagnosticTrailingComment) {
  llvm::StringLiteral testcase = R"(
    // Hello!
    var String x; // trailing comment
  )";

  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TrailingComment,
                            DiagnosticLevel::Error, 3, 19, _)));
  Lex(testcase, consumer);
}

TEST_F(LexerTest, DiagnosticWhitespace) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::NoWhitespaceAfterCommentIntroducer,
                            DiagnosticLevel::Error, 1, 3, _)));
  Lex("//no space after comment", consumer);
}

TEST_F(LexerTest, DiagnosticUnrecognizedEscape) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::UnknownEscapeSequence,
                            DiagnosticLevel::Error, 1, 8, HasSubstr("`b`"))));
  Lex(R"("hello\bworld")", consumer);
}

TEST_F(LexerTest, DiagnosticBadHex) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::HexadecimalEscapeMissingDigits,
                            DiagnosticLevel::Error, 1, 9, _)));
  Lex(R"("hello\xabworld")", consumer);
}

TEST_F(LexerTest, DiagnosticInvalidDigit) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::InvalidDigit,
                            DiagnosticLevel::Error, 1, 6, HasSubstr("'a'"))));
  Lex("0x123abc", consumer);
}

TEST_F(LexerTest, DiagnosticMissingTerminator) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::UnterminatedString,
                            DiagnosticLevel::Error, 1, 1, _)));
  Lex(R"(#" ")", consumer);
}

TEST_F(LexerTest, DiagnosticUnrecognizedChar) {
  Testing::MockDiagnosticConsumer consumer;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::UnrecognizedCharacters,
                            DiagnosticLevel::Error, 1, 1, _)));
  Lex("\b", consumer);
}

TEST_F(LexerTest, PrintingOutputYaml) {
  // Test that we can parse this into YAML and verify line and indent data.
  auto buffer = Lex("\n ;\n\n\n; ;\n\n\n\n\n\n\n\n\n\n\n");
  ASSERT_FALSE(buffer.has_errors());
  TestRawOstream print_stream;
  buffer.Print(print_stream);

  EXPECT_THAT(
      Yaml::Value::FromText(print_stream.TakeStr()),
      IsYaml(ElementsAre(Yaml::Sequence(ElementsAre(Yaml::Mapping(ElementsAre(
          Pair("filename", source_storage_.front().filename().str()),
          Pair("tokens",
               Yaml::Sequence(ElementsAre(
                   Yaml::Mapping(ElementsAre(
                       Pair("index", "0"), Pair("kind", "FileStart"),
                       Pair("line", "1"), Pair("column", "1"),
                       Pair("indent", "1"), Pair("spelling", ""),
                       Pair("has_trailing_space", "true"))),
                   Yaml::Mapping(
                       ElementsAre(Pair("index", "1"), Pair("kind", "Semi"),
                                   Pair("line", "2"), Pair("column", "2"),
                                   Pair("indent", "2"), Pair("spelling", ";"),
                                   Pair("has_trailing_space", "true"))),
                   Yaml::Mapping(
                       ElementsAre(Pair("index", "2"), Pair("kind", "Semi"),
                                   Pair("line", "5"), Pair("column", "1"),
                                   Pair("indent", "1"), Pair("spelling", ";"),
                                   Pair("has_trailing_space", "true"))),
                   Yaml::Mapping(
                       ElementsAre(Pair("index", "3"), Pair("kind", "Semi"),
                                   Pair("line", "5"), Pair("column", "3"),
                                   Pair("indent", "1"), Pair("spelling", ";"),
                                   Pair("has_trailing_space", "true"))),
                   Yaml::Mapping(ElementsAre(
                       Pair("index", "4"), Pair("kind", "FileEnd"),
                       Pair("line", "15"), Pair("column", "1"),
                       Pair("indent", "1"), Pair("spelling", "")))))))))))));
}

}  // namespace
}  // namespace Carbon::Lex
