// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PARSER_PRECEDENCE_H_
#define PARSER_PRECEDENCE_H_

#include "lexer/token_kind.h"
#include "llvm/ADT/Optional.h"

namespace Carbon {

enum class Precedence : int8_t {
  // The first operator has lower precedence (binds more loosely) than the
  // second.
  Lower = -1,
  // Precedence levels are incomparable.
  Incomparable = 0,
  // The first operator has higher precedence (binds more tightly) than the
  // second.
  Higher = 1,
};

// Given two operators `$` and `@`, and an expression `a $ b @ c`, how should
// the expression be grouped?
enum class OperatorPriority : int8_t {
  // The left operator has higher precedence: `(a $ b) @ c`.
  LeftFirst,
  // The right operator has higher precedence: `a $ (b @ c)`.
  RightFirst,
  // The expression is ambiguous.
  Ambiguous
};

enum class Associativity { LeftToRight, None, RightToLeft };

// A precedence group associated with an operator or expression.
class PrecedenceGroup {
 private:
  PrecedenceGroup(int8_t level) : level(level) {}

 public:
  // Objects of this type should only be constructed using the static factory
  // functions below.
  PrecedenceGroup() = delete;

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

  // Get the associativity of this precedence group.
  Associativity GetAssociativity() const;

  // Compare the precedence levels for two adjacent operators.
  static auto Compare(PrecedenceGroup a, PrecedenceGroup b) -> Precedence;

  static auto GetPriority(PrecedenceGroup left, PrecedenceGroup right)
      -> OperatorPriority{
    if (left == right) {
      switch (left.GetAssociativity()) {
        case Associativity::LeftToRight:
          return OperatorPriority::LeftFirst;
        case Associativity::RightToLeft:
          return OperatorPriority::RightFirst;
        case Associativity::None:
          return OperatorPriority::Ambiguous;
      }
    }
    switch (Compare(left, right)) {
      case Precedence::Higher:
        return OperatorPriority::LeftFirst;
      case Precedence::Lower:
        return OperatorPriority::RightFirst;
      case Precedence::Incomparable:
        return OperatorPriority::Ambiguous;
    }
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

#endif  // PARSER_PRECEDENCE_H_
