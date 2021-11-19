// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/ast_test_matchers_internal.h"

namespace Carbon {

auto BlockContentsMatcher::MatchAndExplain(
    Nonnull<const AstNode*> node, ::testing::MatchResultListener* out) const
    -> bool {
  const auto* block = llvm::dyn_cast<Block>(node);
  if (block == nullptr) {
    *out << "is not a Block";
    return false;
  }
  *out << "is a Block whose statements collection ";
  return matcher_.MatchAndExplain(block->statements(), out);
}

auto MatchesIntLiteralMatcher::MatchAndExplain(
    const AstNode* node, ::testing::MatchResultListener* listener) const
    -> bool {
  const auto* literal = llvm::dyn_cast<IntLiteral>(node);
  if (literal == nullptr) {
    *listener << "is not an IntLiteral";
    return false;
  }
  bool matched = literal->value() == value_;
  *listener << "is " << (matched ? "" : "not ") << "a literal " << value_;
  return matched;
}

auto BinaryOperatorExpressionMatcher::MatchAndExplain(
    Nonnull<const AstNode*> node, ::testing::MatchResultListener* out) const
    -> bool {
  const auto* op = llvm::dyn_cast<PrimitiveOperatorExpression>(node);
  if (op == nullptr) {
    *out << "which is not a PrimitiveOperatorExpression";
    return false;
  }
  if (op->arguments().size() != 2) {
    *out << "which does not have two operands";
    return false;
  }
  if (op->op() != op_) {
    *out << "whose operator is not " << ToString(op_);
    return false;
  }
  *out << "which is a " << ToString(op_) << " expression whose left operand ";
  bool matched = lhs_.MatchAndExplain(*op->arguments()[0], out);
  *out << " and right operand ";
  if (!rhs_.MatchAndExplain(*op->arguments()[1], out)) {
    matched = false;
  }
  return matched;
}

void BinaryOperatorExpressionMatcher::DescribeToImpl(std::ostream* out,
                                                     bool negated) const {
  *out << "is " << (negated ? "not " : "") << "a " << ToString(op_)
       << " expression whose ";
  *out << "left operand ";
  lhs_.DescribeTo(out);
  *out << " and right operand ";
  rhs_.DescribeTo(out);
}

auto MatchesReturnMatcher::MatchAndExplain(
    const AstNode* node, ::testing::MatchResultListener* listener) const
    -> bool {
  const auto* ret = llvm::dyn_cast<Return>(node);
  if (ret == nullptr) {
    *listener << "which is not a return statement";
    return false;
  }
  *listener << "which is a return statement ";
  if (ret->is_omitted_expression()) {
    *listener << "with no operand";
    return !matcher_.has_value();
  } else if (matcher_.has_value()) {
    *listener << "whose operand ";
    return matcher_->MatchAndExplain(ret->expression(), listener);
  } else {
    *listener << "that has an operand";
    return false;
  }
}

void MatchesReturnMatcher::DescribeToImpl(std::ostream* out,
                                          bool negated) const {
  *out << "is " << (negated ? "not " : "") << "a return statement ";
  if (matcher_.has_value()) {
    *out << "whose operand ";
    matcher_->DescribeTo(out);
  } else {
    *out << "with no operand";
  }
}

namespace {
// Equivalent to llvm::ListSeparator, but not limited to llvm streams.
class ListSeparator {
 public:
  ListSeparator(std::string separator)
      : separator_(std::move(separator)), should_print_(false) {}

  template <typename OStream>
  friend OStream& operator<<(OStream& out, ListSeparator& sep) {
    if (sep.should_print_) {
      out << sep.separator_;
    }
    sep.should_print_ = true;
    return out;
  }

 private:
  std::string separator_;
  bool should_print_;
};
}  // namespace

auto MatchesFunctionDeclarationMatcher::MatchAndExplain(
    const AstNode* node, ::testing::MatchResultListener* listener) const
    -> bool {
  const auto* decl = llvm::dyn_cast<FunctionDeclaration>(node);
  if (decl == nullptr) {
    *listener << "which is not a function declaration";
    return false;
  }
  *listener << "which is a function declaration ";
  ListSeparator sep(", and");
  if (name_matcher_.has_value()) {
    *listener << sep << "whose name ";
    if (!name_matcher_->MatchAndExplain(decl->name(), listener)) {
      // We short-circuit here because if the name doesn't match, that's
      // probably the only information the user cares about.
      return false;
    }
  }
  bool matched = true;
  if (body_matcher_.has_value()) {
    *listener << sep;
    if (!decl->body().has_value()) {
      *listener << "that doesn't have a body";
      matched = false;
    } else {
      *listener << "whose body ";
      if (!body_matcher_->MatchAndExplain(**decl->body(), listener)) {
        matched = false;
      }
    }
  }
  return matched;
}

void MatchesFunctionDeclarationMatcher::DescribeToImpl(std::ostream* out,
                                                       bool negated) const {
  *out << "is " << (negated ? "not " : "") << "a function declaration ";
  ListSeparator sep(", and");
  if (name_matcher_.has_value()) {
    *out << sep << "whose name ";
    name_matcher_->DescribeTo(out);
  }
  if (body_matcher_.has_value()) {
    *out << sep << "whose body ";
    body_matcher_->DescribeTo(out);
  }
}

}  // namespace Carbon
