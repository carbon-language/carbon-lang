// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include <string>

#include "executable_semantics/syntax/paren_contents.h"
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

TEST(ExpressionTest, EmptyAsExpression) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->LineNumber(), 1);
  ASSERT_EQ(expression->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).Fields(), IsEmpty());
}

TEST(ExpressionTest, EmptyAsTuple) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(), IsEmpty());
}

TEST(ExpressionTest, UnaryNoCommaAsExpression) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  auto term = std::make_unique<IntLiteral>(/*line_num=*/2, 42);
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt, .term = term.get()}},
      .has_trailing_comma = false};

  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->LineNumber(), 2);
  ASSERT_EQ(expression->Tag(), Expression::Kind::IntLiteral);
}

TEST(ExpressionTest, UnaryNoCommaAsTuple) {
  auto term = std::make_unique<IntLiteral>(/*line_num=*/2, 42);
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt, .term = term.get()}},
      .has_trailing_comma = false};

  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(),
              ElementsAre(IntFieldNamed("0")));
}

TEST(ExpressionTest, UnaryWithCommaAsExpression) {
  auto term = std::make_unique<IntLiteral>(/*line_num=*/2, 42);
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt, .term = term.get()}},
      .has_trailing_comma = true};

  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->LineNumber(), 1);
  ASSERT_EQ(expression->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).Fields(),
              ElementsAre(IntFieldNamed("0")));
}

TEST(ExpressionTest, UnaryWithCommaAsTuple) {
  auto term = std::make_unique<IntLiteral>(/*line_num=*/2, 42);
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt, .term = term.get()}},
      .has_trailing_comma = true};

  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(),
              ElementsAre(IntFieldNamed("0")));
}

TEST(ExpressionTest, BinaryAsExpression) {
  auto term1 = std::make_unique<IntLiteral>(/*line_num=*/2, 42);
  auto term2 = std::make_unique<IntLiteral>(/*line_num=*/3, 42);
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt, .term = term1.get()},
                   {.name = std::nullopt, .term = term2.get()}},
      .has_trailing_comma = true};

  const Expression* expression =
      ExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(expression->LineNumber(), 1);
  ASSERT_EQ(expression->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*expression).Fields(),
              ElementsAre(IntFieldNamed("0"), IntFieldNamed("1")));
}

TEST(ExpressionTest, BinaryAsTuple) {
  auto term1 = std::make_unique<IntLiteral>(/*line_num=*/2, 42);
  auto term2 = std::make_unique<IntLiteral>(/*line_num=*/3, 42);
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt, .term = term1.get()},
                   {.name = std::nullopt, .term = term2.get()}},
      .has_trailing_comma = true};

  const Expression* tuple =
      TupleExpressionFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  ASSERT_EQ(tuple->Tag(), Expression::Kind::TupleLiteral);
  EXPECT_THAT(cast<TupleLiteral>(*tuple).Fields(),
              ElementsAre(IntFieldNamed("0"), IntFieldNamed("1")));
}

}  // namespace
}  // namespace Carbon
