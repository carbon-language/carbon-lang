// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression.h"

#include "executable_semantics/syntax/paren_contents.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Carbon {
namespace {

using testing::IsEmpty;

TEST(ExpressionTest, EmptyAsExpression) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  const Expression* expression = AsExpression(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  EXPECT_THAT(expression->GetTupleLiteral().fields, IsEmpty());
}

TEST(ExpressionTest, EmptyAsTuple) {
  ParenContents<Expression> contents = {.elements = {},
                                        .has_trailing_comma = false};
  const Expression* tuple = AsTuple(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  EXPECT_EQ(tuple->GetTupleLiteral().fields.size(), 0);
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

  const Expression* expression = AsExpression(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 2);
  ASSERT_EQ(expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ExpressionTest, UnaryNoCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = false};

  const Expression* tuple = AsTuple(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = tuple->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ExpressionTest, UnaryWithCommaAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = true};

  const Expression* expression = AsExpression(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = expression->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ExpressionTest, UnaryWithCommaAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      .has_trailing_comma = true};

  const Expression* tuple = AsTuple(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = tuple->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ExpressionTest, BinaryAsExpression) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)},
                   {.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/3, 42)}},
      .has_trailing_comma = true};

  const Expression* expression = AsExpression(/*line_num=*/1, contents);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = expression->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
  EXPECT_EQ(fields[1].name, "1");
  EXPECT_EQ(fields[1].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ExpressionTest, BinaryAsTuple) {
  ParenContents<Expression> contents = {
      .elements = {{.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/2, 42)},
                   {.name = std::nullopt,
                    .term = Expression::MakeIntLiteral(/*line_num=*/3, 42)}},
      .has_trailing_comma = true};

  const Expression* tuple = AsTuple(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = tuple->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
  EXPECT_EQ(fields[1].name, "1");
  EXPECT_EQ(fields[1].expression->tag(), ExpressionKind::IntLiteral);
}

}  // namespace
}  // namespace Carbon
