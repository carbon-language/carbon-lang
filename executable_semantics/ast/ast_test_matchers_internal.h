// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation details of the functions in ast_test_matchers.h.

#ifndef EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_INTERNAL_H_
#define EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_INTERNAL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ostream>

#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "llvm/Support/Casting.h"

namespace Carbon {
namespace TestingInternal {

// Matches a Block based on its contents.
class BlockContentsMatcher {
 public:
  using is_gtest_matcher = void;

  // Constructs a matcher which matches a Block node whose .statements() matches
  // `matcher`.
  explicit BlockContentsMatcher(
      ::testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher)
      : matcher_(std::move(matcher)) {}

  void DescribeTo(std::ostream* out) const {
    *out << "is a Block whose statements collection ";
    matcher_.DescribeTo(out);
  }

  void DescribeNegationTo(std::ostream* out) const {
    *out << "is not a Block whose statements collection ";
    matcher_.DescribeTo(out);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* out) const -> bool {
    return MatchAndExplain(&node, out);
  }

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* out) const -> bool;

 private:
  testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher_;
};

// Matches an IntLiteral.
class MatchesIntLiteralMatcher {
 public:
  using is_gtest_matcher = void;

  // Constructs a matcher which matches an IntLiteral whose value() is `value`.
  explicit MatchesIntLiteralMatcher(int value) : value_(value) {}

  void DescribeTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/true);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* listener) const -> bool {
    return MatchAndExplain(&node, listener);
  }

  auto MatchAndExplain(const AstNode* node,
                       ::testing::MatchResultListener* listener) const -> bool;

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const {
    *out << "is " << (negated ? "not " : "") << "a literal " << value_;
  }

  int value_;
};

// Matches a PrimitiveOperatorExpression that has two operands.
class BinaryOperatorExpressionMatcher {
 public:
  using is_gtest_matcher = void;

  // Constructs a matcher which matches a PrimitiveOperatorExpression whose
  // operator is `op`, and which has two operands that match `lhs` and `rhs`
  // respectively.
  explicit BinaryOperatorExpressionMatcher(Operator op,
                                           ::testing::Matcher<AstNode> lhs,
                                           ::testing::Matcher<AstNode> rhs)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  void DescribeTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/true);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* out) const -> bool {
    return MatchAndExplain(&node, out);
  }

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* out) const -> bool;

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const;

  Operator op_;
  ::testing::Matcher<AstNode> lhs_;
  ::testing::Matcher<AstNode> rhs_;
};

// Matches a Return node.
class MatchesReturnMatcher {
 public:
  using is_gtest_matcher = void;

  // Constructs a matcher which matches a Return statement that has no operand.
  explicit MatchesReturnMatcher() = default;

  // Constructs a matcher which matches a Return statement that has an explicit
  // operand that matches `matcher`.
  explicit MatchesReturnMatcher(::testing::Matcher<AstNode> matcher)
      : matcher_(std::move(matcher)) {}

  void DescribeTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/true);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* listener) const -> bool {
    return MatchAndExplain(&node, listener);
  }

  auto MatchAndExplain(const AstNode* node,
                       ::testing::MatchResultListener* listener) const -> bool;

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const;

  std::optional<::testing::Matcher<AstNode>> matcher_;
};

// Matches a FunctionDeclaration. See documentation for
// MatchesFunctionDeclaration in ast_test_matchers.h.
class MatchesFunctionDeclarationMatcher {
 public:
  using is_gtest_matcher = void;

  MatchesFunctionDeclarationMatcher() = default;

  auto WithName(::testing::Matcher<std::string> name_matcher)
      -> MatchesFunctionDeclarationMatcher& {
    name_matcher_ = std::move(name_matcher);
    return *this;
  }

  auto WithBody(::testing::Matcher<AstNode> body_matcher)
      -> MatchesFunctionDeclarationMatcher& {
    body_matcher_ = std::move(body_matcher);
    return *this;
  }

  void DescribeTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/true);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* listener) const -> bool {
    return MatchAndExplain(&node, listener);
  }

  auto MatchAndExplain(const AstNode* node,
                       ::testing::MatchResultListener* listener) const -> bool;

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const;

  std::optional<::testing::Matcher<std::string>> name_matcher_;
  std::optional<::testing::Matcher<AstNode>> body_matcher_;
};

}  // namespace TestingInternal
}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_INTERNAL_H_
