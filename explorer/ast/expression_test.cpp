// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/expression.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "explorer/ast/paren_contents.h"
#include "explorer/common/arena.h"
#include "llvm/Support/Casting.h"

namespace Carbon::Testing {
namespace {

using llvm::cast;
using testing::ElementsAre;
using testing::IsEmpty;

// Matches any `IntLiteral`.
MATCHER(IntField, "") { return arg->kind() == ExpressionKind::IntLiteral; }

static auto FakeSourceLoc(int line_num) -> SourceLocation {
  return SourceLocation("<test>", line_num);
}

class ExpressionTest : public ::testing::Test {
 protected:
  Arena arena;
};

TEST_F(ExpressionTest, EmptyAsExpression) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  Nonnull<const Expression*> expression =
      ExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(expression->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).fields(), IsEmpty());
}

TEST_F(ExpressionTest, EmptyAsTuple) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  Nonnull<const Expression*> tuple =
      TupleExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).fields(), IsEmpty());
}

TEST_F(ExpressionTest, UnaryNoCommaAsExpression) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  ParenContents<Expression> contents = {
      .elements = {arena.New<IntLiteral>(FakeSourceLoc(2), 42)},
      .has_trailing_comma = false};

  Nonnull<const Expression*> expression =
      ExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->source_loc(), FakeSourceLoc(2));
  ASSERT_EQ(expression->kind(), ExpressionKind::IntLiteral);
}

TEST_F(ExpressionTest, UnaryNoCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {arena.New<IntLiteral>(FakeSourceLoc(2), 42)},
      .has_trailing_comma = false};

  Nonnull<const Expression*> tuple =
      TupleExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).fields(), ElementsAre(IntField()));
}

TEST_F(ExpressionTest, UnaryWithCommaAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {arena.New<IntLiteral>(FakeSourceLoc(2), 42)},
      .has_trailing_comma = true};

  Nonnull<const Expression*> expression =
      ExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(expression->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).fields(),
              ElementsAre(IntField()));
}

TEST_F(ExpressionTest, UnaryWithCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {arena.New<IntLiteral>(FakeSourceLoc(2), 42)},
      .has_trailing_comma = true};

  Nonnull<const Expression*> tuple =
      TupleExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).fields(), ElementsAre(IntField()));
}

TEST_F(ExpressionTest, BinaryAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {arena.New<IntLiteral>(FakeSourceLoc(2), 42),
                   arena.New<IntLiteral>(FakeSourceLoc(3), 42)},
      .has_trailing_comma = true};

  Nonnull<const Expression*> expression =
      ExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(expression->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).fields(),
              ElementsAre(IntField(), IntField()));
}

TEST_F(ExpressionTest, BinaryAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {arena.New<IntLiteral>(FakeSourceLoc(2), 42),
                   arena.New<IntLiteral>(FakeSourceLoc(3), 42)},
      .has_trailing_comma = true};

  Nonnull<const Expression*> tuple =
      TupleExpressionFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->source_loc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->kind(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).fields(),
              ElementsAre(IntField(), IntField()));
}

}  // namespace
}  // namespace Carbon::Testing
