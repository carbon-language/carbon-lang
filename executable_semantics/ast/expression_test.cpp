// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <string>

#include "executable_semantics/syntax/paren_contents.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Carbon {
namespace {

using testing::ElementsAre;
using testing::IsEmpty;

// Matches a FieldInitializer named `name` whose `expression` is an
// `IntLiteral`
MATCHER_P(IntFieldNamed, name, "") {
  return arg.name == std::string(name) &&
         arg.expression->tag() == ExpressionKind::IntLiteral;
}

TEST(ExpressionTest, EmptyAsExpression) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(expression->GetTupleLiteral().fields, IsEmpty());
}

TEST(ExpressionTest, EmptyAsTuple) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(tuple->GetTupleLiteral().fields, IsEmpty());
}

TEST(ExpressionTest, UnaryNoCommaAsExpression) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = false};

  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 2);
  ASSERT_EQ(expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ExpressionTest, UnaryNoCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = false};

  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(tuple->GetTupleLiteral().fields, ElementsAre(IntFieldNamed("0")));
}

TEST(ExpressionTest, UnaryWithCommaAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = true};

  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(expression->GetTupleLiteral().fields,
              ElementsAre(IntFieldNamed("0")));
}

TEST(ExpressionTest, UnaryWithCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = true};

  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(tuple->GetTupleLiteral().fields, ElementsAre(IntFieldNamed("0")));
}

TEST(ExpressionTest, BinaryAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)},
                   {.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/3, 42)}},
      .has_trailing_comma = true};

  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(expression->GetTupleLiteral().fields,
              ElementsAre(IntFieldNamed("0"), IntFieldNamed("1")));
}

TEST(ExpressionTest, BinaryAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)},
                   {.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/3, 42)}},
      .has_trailing_comma = true};

  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(tuple->GetTupleLiteral().fields,
              ElementsAre(IntFieldNamed("0"), IntFieldNamed("1")));
}

}  // namespace
}  // namespace Carbon
