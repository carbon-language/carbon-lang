// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/precedence.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "toolchain/lex/token_kind.h"

namespace Carbon::Parse {
namespace {

using ::testing::Eq;

TEST(PrecedenceTest, OperatorsAreRecognized) {
  EXPECT_TRUE(PrecedenceGroup::ForLeading(Lex::TokenKind::Minus).has_value());
  EXPECT_TRUE(PrecedenceGroup::ForLeading(Lex::TokenKind::Caret).has_value());
  EXPECT_FALSE(PrecedenceGroup::ForLeading(Lex::TokenKind::Slash).has_value());
  EXPECT_FALSE(
      PrecedenceGroup::ForLeading(Lex::TokenKind::Identifier).has_value());
  EXPECT_FALSE(PrecedenceGroup::ForLeading(Lex::TokenKind::Tilde).has_value());

  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Minus, false).has_value());
  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Caret, false).has_value());
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Tilde, false).has_value());
  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Slash, true).has_value());
  EXPECT_FALSE(PrecedenceGroup::ForTrailing(Lex::TokenKind::Identifier, false)
                   .has_value());

  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Minus, true)->is_binary);
  EXPECT_FALSE(PrecedenceGroup::ForTrailing(Lex::TokenKind::MinusMinus, false)
                   .has_value());
}

TEST(PrecedenceTest, InfixVsPostfix) {
  // A trailing `-` is always binary, even when written with whitespace that
  // suggests it's postfix.
  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Minus, /*infix*/ false)
          ->is_binary);

  // A trailing `*` is interpreted based on context.
  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, true)->is_binary);
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, false)->is_binary);

  // Infix `*` can appear in `+` contexts; postfix `*` cannot.
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Plus, true)->level),
      Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, false)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Plus, true)->level),
      Eq(OperatorPriority::Ambiguous));
}

TEST(PrecedenceTest, Associativity) {
  EXPECT_THAT(
      PrecedenceGroup::ForLeading(Lex::TokenKind::Minus)->GetAssociativity(),
      Eq(Associativity::None));
  EXPECT_THAT(
      PrecedenceGroup::ForLeading(Lex::TokenKind::Star)->GetAssociativity(),
      Eq(Associativity::RightToLeft));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(Lex::TokenKind::Plus, true)
                  ->level.GetAssociativity(),
              Eq(Associativity::LeftToRight));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(Lex::TokenKind::Equal, true)
                  ->level.GetAssociativity(),
              Eq(Associativity::None));
}

TEST(PrecedenceTest, DirectRelations) {
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Plus, true)->level),
      Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Plus, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, true)->level),
      Eq(OperatorPriority::RightFirst));

  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Amp, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Less, true)->level),
      Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Less, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Amp, true)->level),
      Eq(OperatorPriority::RightFirst));
}

TEST(PrecedenceTest, IndirectRelations) {
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Or, true)->level),
      Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Or, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Star, true)->level),
      Eq(OperatorPriority::RightFirst));

  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          *PrecedenceGroup::ForLeading(Lex::TokenKind::Caret),
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Equal, true)->level),
      Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Equal, true)->level,
          *PrecedenceGroup::ForLeading(Lex::TokenKind::Caret)),
      Eq(OperatorPriority::RightFirst));
}

TEST(PrecedenceTest, IncomparableOperators) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(Lex::TokenKind::Caret),
                  *PrecedenceGroup::ForLeading(Lex::TokenKind::Not)),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(Lex::TokenKind::Caret),
                  *PrecedenceGroup::ForLeading(Lex::TokenKind::Minus)),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          *PrecedenceGroup::ForLeading(Lex::TokenKind::Not),
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Amp, true)->level),
      Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Equal, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::PipeEqual, true)->level),
      Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Plus, true)->level,
          PrecedenceGroup::ForTrailing(Lex::TokenKind::Amp, true)->level),
      Eq(OperatorPriority::Ambiguous));
}

}  // namespace
}  // namespace Carbon::Parse
