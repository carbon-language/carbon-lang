// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <string>

#include "executable_semantics/ast/paren_contents.h"
#include "executable_semantics/common/arena.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/Casting.h"

namespace Carbon {
namespace {

using llvm::cast;
using testing::ElementsAre;
using testing::IsEmpty;

// Matches a FieldInitializer named `name` whose `expression` is an
// `IntLiteral`
MATCHER_P(IntFieldNamed, name, "") {
  return arg.name == std::string(name) &&
         arg.expression->Tag() == Expression::Kind::IntLiteral;
}

static auto FakeSourceLoc(int line_num) -> SourceLocation {
  return SourceLocation("<test>", line_num);
}

class ExpressionTest : public ::testing::Test {
 protected:
  Ptr<Arena> GetArena() { return Ptr<Arena>(&arena); }

  Arena arena;
};

TEST_F(ExpressionTest, EmptyAsExpression) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  Ptr<const Expression> expression =
      ExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(expression->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).Fields(), IsEmpty());
}

TEST_F(ExpressionTest, EmptyAsTuple) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  Ptr<const Expression> tuple =
      TupleExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(), IsEmpty());
}

TEST_F(ExpressionTest, UnaryNoCommaAsExpression) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(2), 42)}},
      .has_trailing_comma = false};

  Ptr<const Expression> expression =
      ExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->SourceLoc(), FakeSourceLoc(2));
  ASSERT_EQ(expression->Tag(), Expression::Kind::IntLiteral);
}

TEST_F(ExpressionTest, UnaryNoCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(2), 42)}},
      .has_trailing_comma = false};

  Ptr<const Expression> tuple =
      TupleExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(),
              ElementsAre(IntFieldNamed("0")));
}

TEST_F(ExpressionTest, UnaryWithCommaAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(2), 42)}},
      .has_trailing_comma = true};

  Ptr<const Expression> expression =
      ExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(expression->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).Fields(),
              ElementsAre(IntFieldNamed("0")));
}

TEST_F(ExpressionTest, UnaryWithCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(2), 42)}},
      .has_trailing_comma = true};

  Ptr<const Expression> tuple =
      TupleExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(),
              ElementsAre(IntFieldNamed("0")));
}

TEST_F(ExpressionTest, BinaryAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(2), 42)},
                   {.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(3), 42)}},
      .has_trailing_comma = true};

  Ptr<const Expression> expression =
      ExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(expression->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(expression->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).Fields(),
              ElementsAre(IntFieldNamed("0"), IntFieldNamed("1")));
}

TEST_F(ExpressionTest, BinaryAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(2), 42)},
                   {.name = std::nullopt,
                    .term = arena.New<IntLiteral>(FakeSourceLoc(3), 42)}},
      .has_trailing_comma = true};

  Ptr<const Expression> tuple =
      TupleExpressionFromParenContents(GetArena(), FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(),
              ElementsAre(IntFieldNamed("0"), IntFieldNamed("1")));
}

}  // namespace
}  // namespace Carbon
