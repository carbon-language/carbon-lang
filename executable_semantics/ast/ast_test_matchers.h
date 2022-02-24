// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Googlemock matchers for the AST. Unless otherwise specified, all the
// functions in this file return matchers that can be applied to any
// AstNode or AstNode*.
//
// TODO: Provide matchers for all node Kinds, and establish more uniform
// conventions for them.

#ifndef EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_H_
#define EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ostream>

#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/ast_test_matchers_internal.h"
#include "executable_semantics/ast/expression.h"

namespace Carbon {

// Matches a Block node whose .statements() match `matcher`.
inline auto BlockContentsAre(
    ::testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher) {
  return TestingInternal::BlockContentsMatcher(std::move(matcher));
}

// Matches a literal with the given value.
// TODO: add overload for string literals
inline auto MatchesLiteral(int value) {
  return TestingInternal::MatchesIntLiteralMatcher(value);
}

// The following functions all match a PrimitiveOperatorExpression with two
// operands that match `lhs` and `rhs` (respectively). The name of the function
// indicates what value of `.op()` they match.
inline auto MatchesMul(::testing::Matcher<AstNode> lhs,
                       ::testing::Matcher<AstNode> rhs) {
  return TestingInternal::BinaryOperatorExpressionMatcher(
      Operator::Mul, std::move(lhs), std::move(rhs));
}

inline auto MatchesAdd(::testing::Matcher<AstNode> lhs,
                       ::testing::Matcher<AstNode> rhs) {
  return TestingInternal::BinaryOperatorExpressionMatcher(
      Operator::Add, std::move(lhs), std::move(rhs));
}

inline auto MatchesAnd(::testing::Matcher<AstNode> lhs,
                       ::testing::Matcher<AstNode> rhs) {
  return TestingInternal::BinaryOperatorExpressionMatcher(
      Operator::And, std::move(lhs), std::move(rhs));
}

inline auto MatchesEq(::testing::Matcher<AstNode> lhs,
                      ::testing::Matcher<AstNode> rhs) {
  return TestingInternal::BinaryOperatorExpressionMatcher(
      Operator::Eq, std::move(lhs), std::move(rhs));
}

inline auto MatchesOr(::testing::Matcher<AstNode> lhs,
                      ::testing::Matcher<AstNode> rhs) {
  return TestingInternal::BinaryOperatorExpressionMatcher(
      Operator::Or, std::move(lhs), std::move(rhs));
}

inline auto MatchesSub(::testing::Matcher<AstNode> lhs,
                       ::testing::Matcher<AstNode> rhs) {
  return TestingInternal::BinaryOperatorExpressionMatcher(
      Operator::Sub, std::move(lhs), std::move(rhs));
}

// Matches a return statement with no operand.
inline auto MatchesEmptyReturn() {
  return TestingInternal::MatchesReturnMatcher();
}

// Matches a return statement with an explicit operand that matches `matcher`.
inline auto MatchesReturn(::testing::Matcher<AstNode> matcher) {
  return TestingInternal::MatchesReturnMatcher(matcher);
}

// Matches a FunctionDeclaration. By default the returned object matches any
// FunctionDeclaration, but it has methods for restricting the match, which can
// be chained fluent-style:
//
// EXPECT_THAT(node, MatchesFunctionDeclaration()
//     .WithName("Foo")
//     .WithBody(BlockContentsAre(...)));
//
// The available methods are:
//
// // *this only matches if the declared name matches name_matcher.
// WithName(::testing::Matcher<std::string> name_matcher)
//
// // *this only matches if the declaration has a body that matches
// // body_matcher.
// WithBody(::testing::Matcher<AstNode> body_matcher)
//
// TODO: Add method for matching only if the declaration has no body.
// TODO: Add methods for matching parameters, deduced parameters,
//   and return term.
inline auto MatchesFunctionDeclaration() {
  return TestingInternal::MatchesFunctionDeclarationMatcher();
}

// Matches an UnimplementedExpression with the given label, whose children
// match `children_matcher`.
inline auto MatchesUnimplementedExpression(
    std::string label,
    ::testing::Matcher<llvm::ArrayRef<Nonnull<const AstNode*>>>
        children_matcher) {
  return TestingInternal::MatchesUnimplementedExpressionMatcher(
      std::move(label), std::move(children_matcher));
}

// Matches an `AST` whose declarations match the given matcher. Unlike other
// matchers in this file, this matcher does not match pointers.
inline auto ASTDeclarations(
    ::testing::Matcher<std::vector<Nonnull<Declaration*>>>
        declarations_matcher) {
  return TestingInternal::ASTDeclarationsMatcher(
      std::move(declarations_matcher));
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_H_
