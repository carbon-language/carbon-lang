// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/precedence.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "toolchain/lexer/token_kind.h"

namespace Carbon::Testing {
namespace {

using ::testing::Eq;

TEST(PrecedenceTest, OperatorsAreRecognized) {
  EXPECT_TRUE(PrecedenceGroup::ForLeading(TokenKind::Minus).has_value());
  EXPECT_TRUE(PrecedenceGroup::ForLeading(TokenKind::Tilde).has_value());
  EXPECT_FALSE(PrecedenceGroup::ForLeading(TokenKind::Slash).has_value());
  EXPECT_FALSE(PrecedenceGroup::ForLeading(TokenKind::Identifier).has_value());

  EXPECT_TRUE(
      PrecedenceGroup::ForTrailing(TokenKind::Minus, false).has_value());
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(TokenKind::Tilde, false).has_value());
  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Slash, true).has_value());
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(TokenKind::Identifier, false).has_value());

  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Minus, true)->is_binary);
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(TokenKind::MinusMinus, false)->is_binary);
}

TEST(PrecedenceTest, InfixVsPostfix) {
  // A trailing `-` is always infix; a trailing `--` is always postfix.
  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Minus, false)->is_binary);
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(TokenKind::MinusMinus, true)->is_binary);

  // A trailing `*` is interpreted based on context.
  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Star, true)->is_binary);
  EXPECT_FALSE(PrecedenceGroup::ForTrailing(TokenKind::Star, false)->is_binary);

  // Infix `*` can appear in type contexts; binary `*` cannot.
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star, true)->level,
                  PrecedenceGroup::ForType()),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star, false)->level,
                  PrecedenceGroup::ForType()),
              Eq(OperatorPriority::LeftFirst));

  // Binary `*` can appear in `+` contexts; binary `*` cannot.
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Plus, true)->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star, false)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Plus, true)->level),
              Eq(OperatorPriority::Ambiguous));
}

TEST(PrecedenceTest, Associativity) {
  EXPECT_THAT(PrecedenceGroup::ForLeading(TokenKind::Minus)->GetAssociativity(),
              Eq(Associativity::RightToLeft));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::PlusPlus, false)
                  ->level.GetAssociativity(),
              Eq(Associativity::LeftToRight));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::Plus, true)
                  ->level.GetAssociativity(),
              Eq(Associativity::LeftToRight));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::Equal, true)
                  ->level.GetAssociativity(),
              Eq(Associativity::RightToLeft));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::PlusEqual, true)
                  ->level.GetAssociativity(),
              Eq(Associativity::None));
}

TEST(PrecedenceTest, DirectRelations) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Plus, true)->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Plus, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Star, true)->level),
              Eq(OperatorPriority::RightFirst));

  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Amp, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Less, true)->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Less, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Amp, true)->level),
              Eq(OperatorPriority::RightFirst));
}

TEST(PrecedenceTest, IndirectRelations) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Or, true)->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Or, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Star, true)->level),
              Eq(OperatorPriority::RightFirst));

  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde),
                  PrecedenceGroup::ForTrailing(TokenKind::Equal, true)->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Equal, true)->level,
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde)),
              Eq(OperatorPriority::RightFirst));
}

TEST(PrecedenceTest, IncomparableOperators) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde),
                  *PrecedenceGroup::ForLeading(TokenKind::Not)),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde),
                  *PrecedenceGroup::ForLeading(TokenKind::Minus)),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Minus),
                  PrecedenceGroup::ForTrailing(TokenKind::Amp, true)->level),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(
      PrecedenceGroup::GetPriority(
          PrecedenceGroup::ForTrailing(TokenKind::Equal, true)->level,
          PrecedenceGroup::ForTrailing(TokenKind::PipeEqual, true)->level),
      Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Plus, true)->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Amp, true)->level),
              Eq(OperatorPriority::Ambiguous));
}

}  // namespace
}  // namespace Carbon::Testing
