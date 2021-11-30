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

// Abstract GoogleMock matcher which matches AstNodes, and is agnostic to
// whether they are passed by pointer or reference. Derived classes specify what
// kinds of AstNodes they match by overriding DescribeToImpl and
// MatchAndExplainImpl.
class AstNodeMatcherBase {
 public:
  using is_gtest_matcher = void;

  virtual ~AstNodeMatcherBase();

  void DescribeTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/true);
  }

  auto MatchAndExplain(const AstNode& node,
                       ::testing::MatchResultListener* out) const -> bool {
    return MatchAndExplainImpl(&node, out);
  }

  auto MatchAndExplain(Nonnull<const AstNode*> node,
                       ::testing::MatchResultListener* out) const -> bool {
    return MatchAndExplainImpl(node, out);
  }

 private:
  // The implementation of this method must satisfy the contract of
  // `DescribeTo(out)` (as specified by GoogleMock) if `negated` is false,
  // or the contract of `DescribeNegationTo(out)` if `negated` is true.
  virtual void DescribeToImpl(std::ostream* out, bool negated) const = 0;

  // The implementation of this method must satisfy the contract of
  // `MatchAndExplain(node, out)`, as specified by GoogleMock.
  virtual auto MatchAndExplainImpl(Nonnull<const AstNode*> node,
                                   ::testing::MatchResultListener* out) const
      -> bool = 0;
};

// Matches a Block based on its contents.
class BlockContentsMatcher : public AstNodeMatcherBase {
 public:
  // Constructs a matcher which matches a Block node whose .statements() matches
  // `matcher`.
  explicit BlockContentsMatcher(
      ::testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher)
      : matcher_(std::move(matcher)) {}

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const override {
    *out << "is " << (negated ? "not " : "")
         << "a Block whose statements collection ";
    matcher_.DescribeTo(out);
  }

  auto MatchAndExplainImpl(Nonnull<const AstNode*> node,
                           ::testing::MatchResultListener* out) const
      -> bool override;

  testing::Matcher<llvm::ArrayRef<Nonnull<const Statement*>>> matcher_;
};

// Matches an IntLiteral.
class MatchesIntLiteralMatcher : public AstNodeMatcherBase {
 public:
  // Constructs a matcher which matches an IntLiteral whose value() is `value`.
  explicit MatchesIntLiteralMatcher(int value) : value_(value) {}

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const override {
    *out << "is " << (negated ? "not " : "") << "a literal " << value_;
  }

  auto MatchAndExplainImpl(const AstNode* node,
                           ::testing::MatchResultListener* listener) const
      -> bool override;

  int value_;
};

// Matches a PrimitiveOperatorExpression that has two operands.
class BinaryOperatorExpressionMatcher : public AstNodeMatcherBase {
 public:
  // Constructs a matcher which matches a PrimitiveOperatorExpression whose
  // operator is `op`, and which has two operands that match `lhs` and `rhs`
  // respectively.
  explicit BinaryOperatorExpressionMatcher(Operator op,
                                           ::testing::Matcher<AstNode> lhs,
                                           ::testing::Matcher<AstNode> rhs)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const override;

  auto MatchAndExplainImpl(Nonnull<const AstNode*> node,
                           ::testing::MatchResultListener* out) const
      -> bool override;

  Operator op_;
  ::testing::Matcher<AstNode> lhs_;
  ::testing::Matcher<AstNode> rhs_;
};

// Matches a Return node.
class MatchesReturnMatcher : public AstNodeMatcherBase {
 public:
  // Constructs a matcher which matches a Return statement that has no operand.
  explicit MatchesReturnMatcher() = default;

  // Constructs a matcher which matches a Return statement that has an explicit
  // operand that matches `matcher`.
  explicit MatchesReturnMatcher(::testing::Matcher<AstNode> matcher)
      : matcher_(std::move(matcher)) {}

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const override;

  auto MatchAndExplainImpl(const AstNode* node,
                           ::testing::MatchResultListener* listener) const
      -> bool override;

  std::optional<::testing::Matcher<AstNode>> matcher_;
};

// Matches a FunctionDeclaration. See documentation for
// MatchesFunctionDeclaration in ast_test_matchers.h.
class MatchesFunctionDeclarationMatcher : public AstNodeMatcherBase {
 public:
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

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const override;
  auto MatchAndExplainImpl(const AstNode* node,
                           ::testing::MatchResultListener* listener) const
      -> bool override;

  std::optional<::testing::Matcher<std::string>> name_matcher_;
  std::optional<::testing::Matcher<AstNode>> body_matcher_;
};

// Matches an UnimplementedExpression.
class MatchesUnimplementedExpressionMatcher : public AstNodeMatcherBase {
 public:
  // Constructs a matcher which matches an UnimplementedExpression that has the
  // given label, and whose children match children_matcher.
  MatchesUnimplementedExpressionMatcher(
      std::string label,
      ::testing::Matcher<llvm::ArrayRef<Nonnull<const AstNode*>>>
          children_matcher)
      : label_(std::move(label)),
        children_matcher_(std::move(children_matcher)) {}

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const override;

  auto MatchAndExplainImpl(Nonnull<const AstNode*> node,
                           ::testing::MatchResultListener* listener) const
      -> bool override;

  std::string label_;
  ::testing::Matcher<llvm::ArrayRef<Nonnull<const AstNode*>>> children_matcher_;
};

// Matches an `AST`.
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
