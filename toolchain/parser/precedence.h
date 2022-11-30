// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PRECEDENCE_H_
#define CARBON_TOOLCHAIN_PARSER_PRECEDENCE_H_

#include <optional>

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
 public:
  struct Trailing;

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
  // return std::nullopt if the given token is not a prefix operator.
  static auto ForLeading(TokenKind kind) -> std::optional<PrecedenceGroup>;

  // Look up the operator information of the given infix or postfix operator
  // token, or return std::nullopt if the given token is not an infix or postfix
  // operator. `infix` indicates whether this is a valid infix operator, but is
  // only considered if the same operator symbol is available as both infix and
  // postfix.
  static auto ForTrailing(TokenKind kind, bool infix)
      -> std::optional<Trailing>;

  friend auto operator==(PrecedenceGroup lhs, PrecedenceGroup rhs) -> bool {
    return lhs.level_ == rhs.level_;
  }
  friend auto operator!=(PrecedenceGroup lhs, PrecedenceGroup rhs) -> bool {
    return lhs.level_ != rhs.level_;
  }

  // Compare the precedence levels for two adjacent operators.
  static auto GetPriority(PrecedenceGroup left, PrecedenceGroup right)
      -> OperatorPriority;

  // Get the associativity of this precedence group.
  [[nodiscard]] auto GetAssociativity() const -> Associativity {
    return static_cast<Associativity>(GetPriority(*this, *this));
  }

 private:
  // We rely on implicit conversions via `int8_t` for enumerators defined in the
  // implementation.
  // NOLINTNEXTLINE(google-explicit-constructor)
  PrecedenceGroup(int8_t level) : level_(level) {}

  // The precedence level.
  int8_t level_;
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

#endif  // CARBON_TOOLCHAIN_PARSER_PRECEDENCE_H_
