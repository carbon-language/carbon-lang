// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation details of the functions in ast_test_matchers.h.

#ifndef EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_INTERNAL_H_
#define EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_INTERNAL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ostream>

#include "executable_semantics/ast/ast.h"
#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "llvm/Support/Casting.h"

namespace Carbon {
namespace TestingInternal {

// Googletest matcher which matches AstNodes according to a specified policy.
// A MatchPolicy must provide two methods:
//
// void DescribeTo(std::ostream* out, bool negated)
// auto MatchAndExplain(Nonnull<const AstNode*> node,
//                      ::testing::MatchResultListener* out) const -> bool
//
// MatchAndExplain has the same requirements as in a GoogleTest matcher.
// DescribeTo has the same requirements as in a GoogleTest matcher when
// `negated` is false, and the same requirements as DescribeNegationTo when
// `negated` is true.
template <typename MatchPolicy>
class AstNodeMatcher {
 public:
  using is_gtest_matcher = void;

  explicit AstNodeMatcher(MatchPolicy policy) : policy_(std::move(policy)) {}

  void DescribeTo(std::ostream* out) const {
    policy_.DescribeTo(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    policy_.DescribeTo(out, /*negated=*/true);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* out) const -> bool {
    return policy_.MatchAndExplain(&node, out);
  }

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* out) const -> bool {
    return policy_.MatchAndExplain(node, out);
  }

 private:
  MatchPolicy policy_;
};

// Explicit deduction guide, to document that CTAD support is intended.
template <typename MatchPolicy>
AstNodeMatcher(MatchPolicy policy) -> AstNodeMatcher<MatchPolicy>;

// Matches a Block based on its contents.
class BlockContentsMatchPolicy {
 public:
  // Constructs a policy which matches a Block node whose .statements() matches
  // `matcher`.
  explicit BlockContentsMatchPolicy(
      ::testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher)
      : matcher_(std::move(matcher)) {}

  void DescribeTo(std::ostream* out, bool negated) const {
    *out << "is " << (negated ? "not " : "")
         << "a Block whose statements collection ";
    matcher_.DescribeTo(out);
  }

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* out) const -> bool;

 private:
  testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher_;
};

// Matches an IntLiteral.
class MatchesIntLiteralPolicy {
 public:
  // Constructs a policy which matches an IntLiteral whose value() is `value`.
  explicit MatchesIntLiteralPolicy(int value) : value_(value) {}

  void DescribeTo(std::ostream* out, bool negated) const {
    *out << "is " << (negated ? "not " : "") << "a literal " << value_;
  }

  auto MatchAndExplain(const AstNode* node,
                       ::testing::MatchResultListener* listener) const -> bool;

 private:
  int value_;
};

// Matches a PrimitiveOperatorExpression that has two operands.
class BinaryOperatorExpressionMatchPolicy {
 public:
  // Constructs a policy which matches a PrimitiveOperatorExpression whose
  // operator is `op`, and which has two operands that match `lhs` and `rhs`
  // respectively.
  explicit BinaryOperatorExpressionMatchPolicy(Operator op,
                                               ::testing::Matcher<AstNode> lhs,
                                               ::testing::Matcher<AstNode> rhs)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  void DescribeTo(std::ostream* out, bool negated) const;

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* out) const -> bool;

 private:
  Operator op_;
  ::testing::Matcher<AstNode> lhs_;
  ::testing::Matcher<AstNode> rhs_;
};

// Matches a Return node.
class MatchesReturnPolicy {
 public:
  // Constructs a policy which matches a Return statement that has no operand.
  explicit MatchesReturnPolicy() = default;

  // Constructs a policy which matches a Return statement that has an explicit
  // operand that matches `matcher`.
  explicit MatchesReturnPolicy(::testing::Matcher<AstNode> matcher)
      : matcher_(std::move(matcher)) {}

  void DescribeTo(std::ostream* out, bool negated) const;

  auto MatchAndExplain(const AstNode* node,
                       ::testing::MatchResultListener* listener) const -> bool;

 private:
  std::optional<::testing::Matcher<AstNode>> matcher_;
};

// Googletest matcher which matches a FunctionDeclaration. See documentation for
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

// Matches an UnimplementedExpression.
class MatchesUnimplementedExpressionPolicy {
 public:
  // Constructs a policy which matches an UnimplementedExpression that has the
  // given label, and whose children match children_matcher.
  MatchesUnimplementedExpressionPolicy(
      std::string label,
      ::testing::Matcher<llvm::ArrayRef<Nonnull<const AstNode*>>>
          children_matcher)
      : label_(std::move(label)),
        children_matcher_(std::move(children_matcher)) {}

  void DescribeTo(std::ostream* out, bool negated) const;

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* listener) const -> bool;

 private:
  std::string label_;
  ::testing::Matcher<llvm::ArrayRef<Nonnull<const AstNode*>>> children_matcher_;
};

// Googletest matcher which matches an `AST`.
class ASTDeclarationsMatcher {
 public:
  using is_gtest_matcher = void;

  // Constructs a matcher which matches an `AST` whose `declarations` member
  // matches `declarations_matcher`
  explicit ASTDeclarationsMatcher(
      ::testing::Matcher<std::vector<Nonnull<Declaration*>>>
          declarations_matcher)
      : declarations_matcher_(std::move(declarations_matcher)) {}

  void DescribeTo(std::ostream* out) const {
    *out << "AST declarations ";
    declarations_matcher_.DescribeTo(out);
  }

  void DescribeNegationTo(std::ostream* out) const {
    *out << "AST declarations ";
    declarations_matcher_.DescribeNegationTo(out);
  }

  auto MatchAndExplain(const AST& ast,
                       ::testing::MatchResultListener* listener) const -> bool {
    *listener << "whose declarations ";
    return declarations_matcher_.MatchAndExplain(ast.declarations, listener);
  }

 private:
  ::testing::Matcher<std::vector<Nonnull<Declaration*>>> declarations_matcher_;
};

}  // namespace TestingInternal
}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_AST_TEST_MATCHERS_INTERNAL_H_
