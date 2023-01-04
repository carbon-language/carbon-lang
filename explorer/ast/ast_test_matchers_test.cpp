// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/ast_test_matchers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/statement.h"
#include "explorer/common/arena.h"

namespace Carbon::Testing {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;

static constexpr SourceLocation DummyLoc("dummy", 0);

TEST(BlockContentsAreTest, BasicUsage) {
  Block empty_block(DummyLoc, {});
  EXPECT_THAT(empty_block, BlockContentsAre(IsEmpty()));
  EXPECT_THAT(&empty_block, BlockContentsAre(IsEmpty()));

  Break break_node(DummyLoc);
  EXPECT_THAT(break_node, Not(BlockContentsAre(_)));

  Block break_block(DummyLoc, {&break_node});
  EXPECT_THAT(break_block, Not(BlockContentsAre(IsEmpty())));
}

TEST(MatchesLiteralTest, BasicUsage) {
  IntLiteral literal(DummyLoc, 42);
  EXPECT_THAT(literal, MatchesLiteral(42));
  EXPECT_THAT(&literal, MatchesLiteral(42));
  EXPECT_THAT(literal, Not(MatchesLiteral(43)));
  EXPECT_THAT(StringLiteral(DummyLoc, "foo"), Not(MatchesLiteral(42)));
}

TEST(MatchesMulTest, BasicUsage) {
  IntLiteral two(DummyLoc, 2);
  IntLiteral three(DummyLoc, 3);
  OperatorExpression mul(DummyLoc, Operator::Mul, {&two, &three});
  EXPECT_THAT(mul, MatchesMul(MatchesLiteral(2), MatchesLiteral(3)));
  EXPECT_THAT(&mul, MatchesMul(MatchesLiteral(2), MatchesLiteral(3)));
  EXPECT_THAT(mul, MatchesMul(_, _));
  EXPECT_THAT(mul, Not(MatchesMul(MatchesLiteral(2), MatchesLiteral(2))));
  EXPECT_THAT(StringLiteral(DummyLoc, "foo"), Not(MatchesMul(_, _)));
  EXPECT_THAT(OperatorExpression(DummyLoc, Operator::Deref, {&two}),
              Not(MatchesMul(_, _)));

  OperatorExpression nested(DummyLoc, Operator::Mul, {&two, &mul});
  EXPECT_THAT(nested,
              MatchesMul(MatchesLiteral(2),
                         MatchesMul(MatchesLiteral(2), MatchesLiteral(3))));
}

TEST(MatchesBinaryOpTest, BasicUsage) {
  IntLiteral two(DummyLoc, 2);
  IntLiteral three(DummyLoc, 3);

  // Testing of MatchesMul provides most of the coverage for these matchers,
  // since they are thin wrappers around a common implementation. We only test
  // the others enough to detect copy-paste errors in the wrappers.
  EXPECT_THAT(OperatorExpression(DummyLoc, Operator::Add, {&two, &three}),
              MatchesAdd(MatchesLiteral(2), MatchesLiteral(3)));
  EXPECT_THAT(OperatorExpression(DummyLoc, Operator::And, {&two, &three}),
              MatchesAnd(MatchesLiteral(2), MatchesLiteral(3)));
  EXPECT_THAT(OperatorExpression(DummyLoc, Operator::Eq, {&two, &three}),
              MatchesEq(MatchesLiteral(2), MatchesLiteral(3)));
  EXPECT_THAT(OperatorExpression(DummyLoc, Operator::Or, {&two, &three}),
              MatchesOr(MatchesLiteral(2), MatchesLiteral(3)));
  EXPECT_THAT(OperatorExpression(DummyLoc, Operator::Sub, {&two, &three}),
              MatchesSub(MatchesLiteral(2), MatchesLiteral(3)));
}

TEST(MatchesReturnTest, BasicUsage) {
  TupleLiteral unit(DummyLoc);
  ReturnExpression empty_return(DummyLoc, &unit,
                                /*is_omitted_expression=*/true);
  EXPECT_THAT(empty_return, MatchesEmptyReturn());
  EXPECT_THAT(&empty_return, MatchesEmptyReturn());
  EXPECT_THAT(empty_return, Not(MatchesReturn(_)));

  IntLiteral int_val(DummyLoc, 42);
  ReturnExpression explicit_return(DummyLoc, &int_val,
                                   /*is_omitted_expression=*/false);
  EXPECT_THAT(explicit_return, MatchesReturn(MatchesLiteral(42)));
  EXPECT_THAT(explicit_return, Not(MatchesEmptyReturn()));

  EXPECT_THAT(int_val, Not(MatchesEmptyReturn()));
  EXPECT_THAT(int_val, Not(MatchesReturn(_)));
}

TEST(MatchesFunctionDeclarationTest, BasicUsage) {
  TuplePattern params(DummyLoc, {});
  Block body(DummyLoc, {});
  FunctionDeclaration decl(DummyLoc, "Foo", {}, std::nullopt, &params,
                           ReturnTerm::Omitted(DummyLoc), &body,
                           VirtualOverride::None);

  EXPECT_THAT(decl, MatchesFunctionDeclaration());
  EXPECT_THAT(&decl, MatchesFunctionDeclaration());
  EXPECT_THAT(decl, MatchesFunctionDeclaration().WithName("Foo"));
  EXPECT_THAT(decl, MatchesFunctionDeclaration().WithBody(_));
  EXPECT_THAT(decl, MatchesFunctionDeclaration().WithName("Foo").WithBody(_));
  EXPECT_THAT(decl, MatchesFunctionDeclaration().WithBody(_).WithName("Foo"));
  EXPECT_THAT(decl, Not(MatchesFunctionDeclaration().WithName("Bar")));
  EXPECT_THAT(decl,
              Not(MatchesFunctionDeclaration().WithBody(MatchesLiteral(0))));

  FunctionDeclaration forward_decl(DummyLoc, "Foo", {}, std::nullopt, &params,
                                   ReturnTerm::Omitted(DummyLoc), std::nullopt,
                                   VirtualOverride::None);
  EXPECT_THAT(forward_decl, MatchesFunctionDeclaration().WithName("Foo"));
  EXPECT_THAT(forward_decl, Not(MatchesFunctionDeclaration().WithBody(_)));

  EXPECT_THAT(body, Not(MatchesFunctionDeclaration()));
}

TEST(MatchesUnimplementedExpressionTest, BasicUsage) {
  IntLiteral two(DummyLoc, 2);
  IntLiteral three(DummyLoc, 3);
  UnimplementedExpression unimplemented(DummyLoc, "DummyLabel", &two, &three);

  EXPECT_THAT(unimplemented, MatchesUnimplementedExpression(
                                 "DummyLabel", ElementsAre(MatchesLiteral(2),
                                                           MatchesLiteral(3))));
  EXPECT_THAT(
      &unimplemented,
      MatchesUnimplementedExpression(
          "DummyLabel", ElementsAre(MatchesLiteral(2), MatchesLiteral(3))));
  EXPECT_THAT(
      unimplemented,
      Not(MatchesUnimplementedExpression(
          "WrongLabel", ElementsAre(MatchesLiteral(2), MatchesLiteral(3)))));
  EXPECT_THAT(unimplemented,
              Not(MatchesUnimplementedExpression("DummyLabel", IsEmpty())));
  EXPECT_THAT(two,
              Not(MatchesUnimplementedExpression("DummyLabel", IsEmpty())));
}

TEST(ASTDeclarationsTest, BasicUsage) {
  TuplePattern params(DummyLoc, {});
  Block body(DummyLoc, {});
  FunctionDeclaration decl(DummyLoc, "Foo", {}, std::nullopt, &params,
                           ReturnTerm::Omitted(DummyLoc), &body,
                           VirtualOverride::None);
  AST ast = {.declarations = {&decl}};

  EXPECT_THAT(ast, ASTDeclarations(ElementsAre(MatchesFunctionDeclaration())));
  EXPECT_THAT(ast, Not(ASTDeclarations(IsEmpty())));
}

}  // namespace
}  // namespace Carbon::Testing
