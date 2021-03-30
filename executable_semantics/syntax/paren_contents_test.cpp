// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/paren_contents.h"

#include "gtest/gtest.h"

namespace Carbon {
namespace {

TEST(ParenContentsTest, EmptyAsExpression) {
  ParenContents contents;
  Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag, ExpressionKind::Tuple);
  EXPECT_EQ(expression->u.tuple.fields->size(), 0);
}

TEST(ParenContentsTest, EmptyAsTuple) {
  ParenContents contents;
  Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag, ExpressionKind::Tuple);
  EXPECT_EQ(tuple->u.tuple.fields->size(), 0);
}

TEST(ParenContentsTest, UnaryNoCommaAsExpression) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  ParenContents contents({{.expression = MakeInt(/*line_num=*/2, 42)}},
                         ParenContents::HasTrailingComma::No);

  Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 2);
  ASSERT_EQ(expression->tag, ExpressionKind::Integer);
}

TEST(ParenContentsTest, UnaryNoCommaAsTuple) {
  ParenContents contents({{.expression = MakeInt(/*line_num=*/2, 42)}},
                         ParenContents::HasTrailingComma::No);

  Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag, ExpressionKind::Tuple);
  std::vector<std::pair<std::string, Expression*>> fields =
      *tuple->u.tuple.fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].second->tag, ExpressionKind::Integer);
}

TEST(ParenContentsTest, UnaryWithCommaAsExpression) {
  ParenContents contents({{.expression = MakeInt(/*line_num=*/2, 42)}},
                         ParenContents::HasTrailingComma::Yes);

  Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag, ExpressionKind::Tuple);
  std::vector<std::pair<std::string, Expression*>> fields =
      *expression->u.tuple.fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].second->tag, ExpressionKind::Integer);
}

TEST(ParenContentsTest, UnaryWithCommaAsTuple) {
  ParenContents contents({{.expression = MakeInt(/*line_num=*/2, 42)}},
                         ParenContents::HasTrailingComma::Yes);

  Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag, ExpressionKind::Tuple);
  std::vector<std::pair<std::string, Expression*>> fields =
      *tuple->u.tuple.fields;
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].second->tag, ExpressionKind::Integer);
}

TEST(ParenContentsTest, BinaryAsExpression) {
  ParenContents contents({{.expression = MakeInt(/*line_num=*/2, 42)},
                          {.expression = MakeInt(/*line_num=*/3, 42)}},
                         ParenContents::HasTrailingComma::Yes);

  Expression* expression = contents.AsExpression(/*line_num=*/1);
  EXPECT_EQ(expression->line_num, 1);
  ASSERT_EQ(expression->tag, ExpressionKind::Tuple);
  std::vector<std::pair<std::string, Expression*>> fields =
      *expression->u.tuple.fields;
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].second->tag, ExpressionKind::Integer);
  EXPECT_EQ(fields[1].second->tag, ExpressionKind::Integer);
}

TEST(ParenContentsTest, BinaryAsTuple) {
  ParenContents contents({{.expression = MakeInt(/*line_num=*/2, 42)},
                          {.expression = MakeInt(/*line_num=*/3, 42)}},
                         ParenContents::HasTrailingComma::Yes);

  Expression* tuple = contents.AsTuple(/*line_num=*/1);
  EXPECT_EQ(tuple->line_num, 1);
  ASSERT_EQ(tuple->tag, ExpressionKind::Tuple);
  std::vector<std::pair<std::string, Expression*>> fields =
      *tuple->u.tuple.fields;
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].second->tag, ExpressionKind::Integer);
  EXPECT_EQ(fields[1].second->tag, ExpressionKind::Integer);
}

}  // namespace
}  // namespace Carbon
