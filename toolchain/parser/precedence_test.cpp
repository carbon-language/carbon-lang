// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/precedence.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "toolchain/lexer/token_kind.h"

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::Ne;

TEST(PrecedenceTest, OperatorsAreRecognized) {
  EXPECT_TRUE(PrecedenceGroup::ForLeading(TokenKind::Minus()).hasValue());
  EXPECT_TRUE(PrecedenceGroup::ForLeading(TokenKind::Tilde()).hasValue());
  EXPECT_FALSE(PrecedenceGroup::ForLeading(TokenKind::Slash()).hasValue());
  EXPECT_FALSE(PrecedenceGroup::ForLeading(TokenKind::Identifier()).hasValue());

  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Minus()).hasValue());
  EXPECT_FALSE(PrecedenceGroup::ForTrailing(TokenKind::Tilde()).hasValue());
  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Slash()).hasValue());
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(TokenKind::Identifier()).hasValue());

  EXPECT_TRUE(PrecedenceGroup::ForTrailing(TokenKind::Minus())->is_binary);
  EXPECT_FALSE(
      PrecedenceGroup::ForTrailing(TokenKind::MinusMinus())->is_binary);
}

TEST(PrecedenceTest, Associativity) {
  EXPECT_THAT(
      PrecedenceGroup::ForLeading(TokenKind::Minus())->GetAssociativity(),
      Eq(Associativity::RightToLeft));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::PlusPlus())
                  ->level.GetAssociativity(),
              Eq(Associativity::LeftToRight));
  EXPECT_THAT(
      PrecedenceGroup::ForTrailing(TokenKind::Plus())->level.GetAssociativity(),
      Eq(Associativity::LeftToRight));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::Equal())
                  ->level.GetAssociativity(),
              Eq(Associativity::RightToLeft));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::PlusEqual())
                  ->level.GetAssociativity(),
              Eq(Associativity::None));
}

TEST(PrecedenceTest, DirectRelations) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Plus())->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Plus())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level),
              Eq(OperatorPriority::RightFirst));

  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Less())->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Less())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level),
              Eq(OperatorPriority::RightFirst));
}

TEST(PrecedenceTest, IndirectRelations) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::OrKeyword())->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::OrKeyword())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level),
              Eq(OperatorPriority::RightFirst));

  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde()),
                  PrecedenceGroup::ForTrailing(TokenKind::Equal())->level),
              Eq(OperatorPriority::LeftFirst));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Equal())->level,
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde())),
              Eq(OperatorPriority::RightFirst));
}

TEST(PrecedenceTest, IncomparableOperators) {
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde()),
                  *PrecedenceGroup::ForLeading(TokenKind::NotKeyword())),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde()),
                  *PrecedenceGroup::ForLeading(TokenKind::Minus())),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  *PrecedenceGroup::ForLeading(TokenKind::Minus()),
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Equal())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::PipeEqual())->level),
              Eq(OperatorPriority::Ambiguous));
  EXPECT_THAT(PrecedenceGroup::GetPriority(
                  PrecedenceGroup::ForTrailing(TokenKind::Plus())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level),
              Eq(OperatorPriority::Ambiguous));
}

}  // namespace
}  // namespace Carbon
