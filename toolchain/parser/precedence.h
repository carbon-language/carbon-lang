// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_PARSER_PRECEDENCE_H_
#define TOOLCHAIN_PARSER_PRECEDENCE_H_

#include "llvm/ADT/Optional.h"
#include "toolchain/lexer/token_kind.h"

namespace Carbon {

// Given two operators `$` and `@`, and an expression `a $ b @ c`, how should
// the expression be parsed?
enum class OperatorPriority : int8_t {
  // The left operator has higher precedence: `(a $ b) @ c`.
  LeftFirst = -1,
  // The expression is ambiguous.
  Ambiguous = 0,
  // The right operator has higher precedence: `a $ (b @ c)`.
  RightFirst = 1,
};

enum class Associativity : int8_t {
  LeftToRight = -1,
  None = 0,
  RightToLeft = 1
};

// A precedence group associated with an operator or expression.
class PrecedenceGroup {
 private:
  PrecedenceGroup(int8_t level) : level(level) {}

 public:
  // Objects of this type should only be constructed using the static factory
  // functions below.
  PrecedenceGroup() = delete;

  // Get the sentinel precedence level for a postfix expression. All operators
  // should have lower precedence than this.
  static auto ForPostfixExpression() -> PrecedenceGroup;

  // Get the sentinel precedence level for a top-level expression context. All
  // operators should have higher precedence than this.
  static auto ForTopLevelExpression() -> PrecedenceGroup;

  // Get the precedence level at which to parse a type expression. All type
  // operators should have higher precedence than this.
  static auto ForType() -> PrecedenceGroup;

  // Look up the operator information of the given prefix operator token, or
  // return llvm::None if the given token is not a prefix operator.
  static auto ForLeading(TokenKind kind) -> llvm::Optional<PrecedenceGroup>;

  struct Trailing;

  // Look up the operator information of the given infix or postfix operator
  // token, or return llvm::None if the given token is not an infix or postfix
  // operator.
  static auto ForTrailing(TokenKind kind) -> llvm::Optional<Trailing>;

  friend auto operator==(PrecedenceGroup lhs, PrecedenceGroup rhs) -> bool {
    return lhs.level == rhs.level;
  }
  friend auto operator!=(PrecedenceGroup lhs, PrecedenceGroup rhs) -> bool {
    return lhs.level != rhs.level;
  }

  // Compare the precedence levels for two adjacent operators.
  static auto GetPriority(PrecedenceGroup left, PrecedenceGroup right)
      -> OperatorPriority;

  // Get the associativity of this precedence group.
  Associativity GetAssociativity() const {
    return static_cast<Associativity>(GetPriority(*this, *this));
  }

 private:
  // The precedence level.
  int8_t level;
};

// Precedence information for a trailing operator.
struct PrecedenceGroup::Trailing {
  // The precedence level.
  PrecedenceGroup level;
  // `true` if this is an infix binary operator, `false` if this is a postfix
  // unary operator.
  bool is_binary;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_PARSER_PRECEDENCE_H_
