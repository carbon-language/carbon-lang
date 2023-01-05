// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/token_kind.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>

#include "llvm/ADT/StringRef.h"

namespace Carbon::Testing {
namespace {

using ::testing::MatchesRegex;

// We restrict symbols to punctuation characters that are expected to be widely
// available on modern keyboards used for programming.
constexpr llvm::StringLiteral SymbolRegex =
    R"([\[\]{}!@#%^&*()/?\\|;:.,<>=+~-]+)";

// We restrict keywords to be lowercase ASCII letters and underscores with one
// exception: `Self`.
constexpr llvm::StringLiteral KeywordRegex = "[a-z_]+|Self";

#define CARBON_TOKEN(TokenName)                           \
  TEST(TokenKindTest, TokenName) {                        \
    EXPECT_EQ(#TokenName, TokenKind::TokenName.name());   \
    EXPECT_FALSE(TokenKind::TokenName.is_symbol());       \
    EXPECT_FALSE(TokenKind::TokenName.is_keyword());      \
    EXPECT_EQ("", TokenKind::TokenName.fixed_spelling()); \
  }
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling)                \
  TEST(TokenKindTest, TokenName) {                              \
    EXPECT_EQ(#TokenName, TokenKind::TokenName.name());         \
    EXPECT_TRUE(TokenKind::TokenName.is_symbol());              \
    EXPECT_FALSE(TokenKind::TokenName.is_grouping_symbol());    \
    EXPECT_FALSE(TokenKind::TokenName.is_opening_symbol());     \
    EXPECT_FALSE(TokenKind::TokenName.is_closing_symbol());     \
    EXPECT_FALSE(TokenKind::TokenName.is_keyword());            \
    EXPECT_EQ(Spelling, TokenKind::TokenName.fixed_spelling()); \
    EXPECT_THAT(Spelling, MatchesRegex(SymbolRegex.str()));     \
  }
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName)   \
  TEST(TokenKindTest, TokenName) {                                            \
    EXPECT_EQ(#TokenName, TokenKind::TokenName.name());                       \
    EXPECT_TRUE(TokenKind::TokenName.is_symbol());                            \
    EXPECT_TRUE(TokenKind::TokenName.is_grouping_symbol());                   \
    EXPECT_TRUE(TokenKind::TokenName.is_opening_symbol());                    \
    EXPECT_EQ(TokenKind::ClosingName, TokenKind::TokenName.closing_symbol()); \
    EXPECT_FALSE(TokenKind::TokenName.is_closing_symbol());                   \
    EXPECT_FALSE(TokenKind::TokenName.is_keyword());                          \
    EXPECT_EQ(Spelling, TokenKind::TokenName.fixed_spelling());               \
    EXPECT_THAT(Spelling, MatchesRegex(SymbolRegex.str()));                   \
  }
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName)   \
  TEST(TokenKindTest, TokenName) {                                            \
    EXPECT_EQ(#TokenName, TokenKind::TokenName.name());                       \
    EXPECT_TRUE(TokenKind::TokenName.is_symbol());                            \
    EXPECT_TRUE(TokenKind::TokenName.is_grouping_symbol());                   \
    EXPECT_FALSE(TokenKind::TokenName.is_opening_symbol());                   \
    EXPECT_TRUE(TokenKind::TokenName.is_closing_symbol());                    \
    EXPECT_EQ(TokenKind::OpeningName, TokenKind::TokenName.opening_symbol()); \
    EXPECT_FALSE(TokenKind::TokenName.is_keyword());                          \
    EXPECT_EQ(Spelling, TokenKind::TokenName.fixed_spelling());               \
    EXPECT_THAT(Spelling, MatchesRegex(SymbolRegex.str()));                   \
  }
#define CARBON_KEYWORD_TOKEN(TokenName, Spelling)               \
  TEST(TokenKindTest, TokenName) {                              \
    EXPECT_EQ(#TokenName, TokenKind::TokenName.name());         \
    EXPECT_FALSE(TokenKind::TokenName.is_symbol());             \
    EXPECT_TRUE(TokenKind::TokenName.is_keyword());             \
    EXPECT_EQ(Spelling, TokenKind::TokenName.fixed_spelling()); \
    EXPECT_THAT(Spelling, MatchesRegex(KeywordRegex.str()));    \
  }
#include "toolchain/lexer/token_kind.def"

// Verify that the symbol tokens are sorted from longest to shortest. This is
// important to ensure that simply in-order testing will identify tokens
// following the max-munch rule.
TEST(TokenKindTest, SymbolsInDescendingLength) {
  int previous_length = INT_MAX;
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling)                        \
  EXPECT_LE(llvm::StringRef(Spelling).size(), previous_length)          \
      << "Symbol token not in descending length order: " << #TokenName; \
  previous_length = llvm::StringRef(Spelling).size();
#include "toolchain/lexer/token_kind.def"
  EXPECT_GT(previous_length, 0);
}

}  // namespace
}  // namespace Carbon::Testing
