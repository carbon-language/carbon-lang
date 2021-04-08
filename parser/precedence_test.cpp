// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "parser/precedence.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lexer/token_kind.h"

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::Ne;

struct PrecedenceTest : ::testing::Test {};

TEST_F(PrecedenceTest, OperatorsAreRecognized) {
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

TEST_F(PrecedenceTest, Associativity) {
  EXPECT_THAT(
      PrecedenceGroup::ForLeading(TokenKind::Minus())->GetAssociativity(),
      Eq(Associativity::LeftToRight));
  EXPECT_THAT(PrecedenceGroup::ForTrailing(TokenKind::PlusPlus())
                  ->level.GetAssociativity(),
              Eq(Associativity::RightToLeft));
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

TEST_F(PrecedenceTest, DirectRelations) {
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Plus())->level),
              Eq(Precedence::Higher));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Plus())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level),
              Eq(Precedence::Lower));

  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Less())->level),
              Eq(Precedence::Higher));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Less())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level),
              Eq(Precedence::Lower));
}

TEST_F(PrecedenceTest, IndirectRelations) {
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::OrKeyword())->level),
              Eq(Precedence::Higher));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::OrKeyword())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Star())->level),
              Eq(Precedence::Lower));

  EXPECT_THAT(PrecedenceGroup::Compare(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde()),
                  PrecedenceGroup::ForTrailing(TokenKind::Equal())->level),
              Eq(Precedence::Higher));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Equal())->level,
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde())),
              Eq(Precedence::Lower));
}

TEST_F(PrecedenceTest, IncomparableOperators) {
  EXPECT_THAT(PrecedenceGroup::Compare(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde()),
                  *PrecedenceGroup::ForLeading(TokenKind::NotKeyword())),
              Eq(Precedence::Incomparable));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  *PrecedenceGroup::ForLeading(TokenKind::Tilde()),
                  *PrecedenceGroup::ForLeading(TokenKind::Minus())),
              Eq(Precedence::Incomparable));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  *PrecedenceGroup::ForLeading(TokenKind::Minus()),
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level),
              Eq(Precedence::Incomparable));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Equal())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::PipeEqual())->level),
              Eq(Precedence::Incomparable));
  EXPECT_THAT(PrecedenceGroup::Compare(
                  PrecedenceGroup::ForTrailing(TokenKind::Plus())->level,
                  PrecedenceGroup::ForTrailing(TokenKind::Amp())->level),
              Eq(Precedence::Incomparable));
}

}  // namespace
}  // namespace Carbon
