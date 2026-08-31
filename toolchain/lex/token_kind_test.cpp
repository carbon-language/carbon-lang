// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"

namespace Carbon::Lex {
namespace {

using ::testing::MatchesRegex;

// We restrict symbols to punctuation characters that are expected to be widely
// available on modern keyboards used for programming.
constexpr llvm::StringLiteral SymbolRegex =
    R"([\[\]{}!@#%^&*()/?\\|;:.,<>=+~-]+)";

// We restrict keywords to be lowercase ASCII letters and underscores with a few
// specific exceptions.
constexpr llvm::StringLiteral KeywordRegex = "[a-z_]+|Core|Cpp|Self";

static void CheckToken(TokenKind kind) {
  EXPECT_FALSE(kind.is_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_keyword()) << kind.name().str();
  EXPECT_EQ("", kind.fixed_spelling()) << kind.name().str();
}

static void CheckSymbolToken(TokenKind kind, llvm::StringRef spelling) {
  EXPECT_TRUE(kind.is_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_grouping_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_opening_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_closing_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_keyword()) << kind.name().str();
  EXPECT_EQ(spelling, kind.fixed_spelling()) << kind.name().str();
  EXPECT_THAT(spelling.str(), MatchesRegex(SymbolRegex.str()))
      << kind.name().str();
}

static void CheckOpeningGroupSymbolToken(TokenKind kind,
                                         llvm::StringRef spelling,
                                         TokenKind closing_kind) {
  EXPECT_TRUE(kind.is_symbol()) << kind.name().str();
  EXPECT_TRUE(kind.is_grouping_symbol()) << kind.name().str();
  EXPECT_TRUE(kind.is_opening_symbol()) << kind.name().str();
  EXPECT_EQ(closing_kind, kind.closing_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_closing_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_keyword()) << kind.name().str();
  EXPECT_EQ(spelling, kind.fixed_spelling()) << kind.name().str();
  EXPECT_THAT(spelling.str(), MatchesRegex(SymbolRegex.str()))
      << kind.name().str();
}

static void CheckClosingGroupSymbolToken(TokenKind kind,
                                         llvm::StringRef spelling,
                                         TokenKind opening_kind) {
  EXPECT_TRUE(kind.is_symbol()) << kind.name().str();
  EXPECT_TRUE(kind.is_grouping_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_opening_symbol()) << kind.name().str();
  EXPECT_TRUE(kind.is_closing_symbol()) << kind.name().str();
  EXPECT_EQ(opening_kind, kind.opening_symbol()) << kind.name().str();
  EXPECT_FALSE(kind.is_keyword()) << kind.name().str();
  EXPECT_EQ(spelling, kind.fixed_spelling()) << kind.name().str();
  EXPECT_THAT(spelling.str(), MatchesRegex(SymbolRegex.str()))
      << kind.name().str();
}

static void CheckKeywordToken(TokenKind kind, llvm::StringRef spelling) {
  EXPECT_FALSE(kind.is_symbol()) << kind.name().str();
  EXPECT_TRUE(kind.is_keyword()) << kind.name().str();
  EXPECT_EQ(spelling, kind.fixed_spelling()) << kind.name().str();
  EXPECT_THAT(spelling.str(), MatchesRegex(KeywordRegex.str()))
      << kind.name().str();
}

TEST(TokenKindTest, AllTokens) {
#define CARBON_TOKEN(TokenName) CheckToken(TokenKind::TokenName);
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) \
  CheckSymbolToken(TokenKind::TokenName, Spelling);
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  CheckOpeningGroupSymbolToken(TokenKind::TokenName, Spelling,              \
                               TokenKind::ClosingName);
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  CheckClosingGroupSymbolToken(TokenKind::TokenName, Spelling,              \
                               TokenKind::OpeningName);
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling) \
  CheckKeywordToken(TokenKind::TokenName, Spelling);
#include "toolchain/lex/token_kind.def"
}

// Verify that the symbol tokens are sorted from longest to shortest. This is
// important to ensure that simply in-order testing will identify tokens
// following the max-munch rule.
TEST(TokenKindTest, SymbolsInDescendingLength) {
  int previous_length = INT_MAX;
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling)                        \
  EXPECT_LE(llvm::StringRef(Spelling).size(), previous_length)          \
      << "Symbol token not in descending length order: " << #TokenName; \
  previous_length = llvm::StringRef(Spelling).size();
#include "toolchain/lex/token_kind.def"
  EXPECT_GT(previous_length, 0);
}

}  // namespace
}  // namespace Carbon::Lex
