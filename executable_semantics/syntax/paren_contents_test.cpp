// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/paren_contents.h"

#include "gtest/gtest.h"

namespace Carbon {
namespace {

TEST(ParenContentsTest, EmptyAsExpression) {
  ParenContents contents;
  const Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  EXPECT_EQ(expression->GetTupleLiteral().fields.size(), 0);
}

TEST(ParenContentsTest, EmptyAsTuple) {
  ParenContents contents;
  const Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  EXPECT_EQ(tuple->GetTupleLiteral().fields.size(), 0);
}

TEST(ParenContentsTest, UnaryNoCommaAsExpression) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  ParenContents contents(
      {{.expression = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      ParenContents::HasTrailingComma::No);

  const Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 2);
  ASSERT_EQ(expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ParenContentsTest, UnaryNoCommaAsTuple) {
  ParenContents contents(
      {{.expression = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      ParenContents::HasTrailingComma::No);

  const Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = tuple->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ParenContentsTest, UnaryWithCommaAsExpression) {
  ParenContents contents(
      {{.expression = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      ParenContents::HasTrailingComma::Yes);

  const Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = expression->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ParenContentsTest, UnaryWithCommaAsTuple) {
  ParenContents contents(
      {{.expression = Expression::MakeIntLiteral(/*line_num=*/2, 42)}},
      ParenContents::HasTrailingComma::Yes);

  const Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = tuple->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ParenContentsTest, BinaryAsExpression) {
  ParenContents contents(
      {{.expression = Expression::MakeIntLiteral(/*line_num=*/2, 42)},
       {.expression = Expression::MakeIntLiteral(/*line_num=*/3, 42)}},
      ParenContents::HasTrailingComma::Yes);

  const Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = expression->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
  EXPECT_EQ(fields[1].expression->tag(), ExpressionKind::IntLiteral);
}

TEST(ParenContentsTest, BinaryAsTuple) {
  ParenContents contents(
      {{.expression = Expression::MakeIntLiteral(/*line_num=*/2, 42)},
       {.expression = Expression::MakeIntLiteral(/*line_num=*/3, 42)}},
      ParenContents::HasTrailingComma::Yes);

  const Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag(), ExpressionKind::TupleLiteral);
  std::vector<FieldInitializer> fields = tuple->GetTupleLiteral().fields;
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].expression->tag(), ExpressionKind::IntLiteral);
  EXPECT_EQ(fields[1].expression->tag(), ExpressionKind::IntLiteral);
}

}  // namespace
}  // namespace Carbon
