// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "parser/precedence.h"

#include <utility>

namespace Carbon {

namespace {
enum PrecedenceLevel : int8_t {
  // Numeric.
  NumericPrefix,
  NumericPostfix,
  Modulo,
  Multiplicative,
  Additive,
  // Bitwise.
  BitwisePrefix,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  BitShift,
  // Logical.
  LogicalPrefix,
  Relational,
  LogicalAnd,
  LogicalOr,
  // Assignment.
  SimpleAssignment,
  CompoundAssignment,
};
constexpr int8_t NumPrecedenceLevels = CompoundAssignment + 1;

// A precomputed lookup table determining the relative precedence of two
// precedence groups.
struct OperatorPriorityTable {
  constexpr OperatorPriorityTable() : table{} {
    // Start with a list of <higher precedence>, <lower precedence>
    // relationships.
    MarkHigherThan({NumericPrefix, NumericPostfix},
                   {Modulo, Multiplicative, BitShift});
    MarkHigherThan({Multiplicative}, {Additive});
    MarkHigherThan({BitwisePrefix},
                   {BitwiseAnd, BitwiseOr, BitwiseXor, BitShift});
    MarkHigherThan(
        {Modulo, Additive, BitwiseAnd, BitwiseOr, BitwiseXor, BitShift},
        {SimpleAssignment, CompoundAssignment, Relational});
    MarkHigherThan({Relational, LogicalPrefix}, {LogicalAnd, LogicalOr});

    // Compute the transitive closure of the above relationships: if we parse
    // `a $ b @ c` as `(a $ b) @ c` and parse `b @ c % d` as `(b @ c) % d`,
    // then we will parse `a $ b @ c % d` as `((a $ b) @ c) % d` and should
    // also parse `a $ bc % d` as `(a $ bc) % d`.
    MakeTransitivelyClosed();

    // Make the relation symmetric. If we parse `a $ b @ c` as `(a $ b) @ c`
    // then we want to parse `a @ b $ c` as `a @ (b $ c)`.
    MakeSymmetric();

    // Fill in the diagonal, which represents operator associativity.
    AddAssociativityRules();
  }

  constexpr void MarkHigherThan(
      std::initializer_list<PrecedenceLevel> higher_group,
      std::initializer_list<PrecedenceLevel> lower_group) {
    for (auto higher : higher_group) {
      for (auto lower : lower_group) {
        table[higher][lower] = OperatorPriority::LeftFirst;
      }
    }
  }

  constexpr void MakeTransitivelyClosed() {
    // A naive algorithm compiles acceptably fast for now (~0.5s). This should
    // be revisited if we see compile time problems after adding precedence
    // groups; it's easy to do this faster.
    bool changed = false;
    do {
      changed = false;
      for (int8_t a = 0; a != NumPrecedenceLevels; ++a) {
        for (int8_t b = 0; b != NumPrecedenceLevels; ++b) {
          if (table[a][b] == OperatorPriority::LeftFirst) {
            for (int8_t c = 0; c != NumPrecedenceLevels; ++c) {
              if (table[b][c] == OperatorPriority::LeftFirst &&
                  table[a][c] != OperatorPriority::LeftFirst) {
                table[a][c] = OperatorPriority::LeftFirst;
                changed = true;
              }
            }
          }
        }
      }
    } while (changed);
  }

  constexpr void MakeSymmetric() {
    for (int8_t a = 0; a != NumPrecedenceLevels; ++a) {
      for (int8_t b = 0; b != NumPrecedenceLevels; ++b) {
        if (table[a][b] == OperatorPriority::LeftFirst) {
          if (table[b][a] == OperatorPriority::LeftFirst) {
            throw "inconsistent lookup table entries";
          }
          table[b][a] = OperatorPriority::RightFirst;
        }
      }
    }
  }

  constexpr void AddAssociativityRules() {
    // Associativity rules occupy the diagonal

    // For prefix operators, RightFirst would mean `@@x` is `@(@x)` and
    // Ambiguous would mean it's an error. LeftFirst is meaningless. For now we
    // allow all prefix operators to be repeated.
    for (PrecedenceLevel prefix :
         {NumericPrefix, BitwisePrefix, LogicalPrefix}) {
      table[prefix][prefix] = OperatorPriority::RightFirst;
    }

    // Postfix operators are symmetric with prefix operators.
    for (PrecedenceLevel postfix : {NumericPostfix}) {
      table[postfix][postfix] = OperatorPriority::LeftFirst;
    }

    // Traditionally-associative operators are given left-to-right
    // associativity.
    for (PrecedenceLevel assoc :
         {Multiplicative, Additive, BitwiseAnd, BitwiseOr, BitwiseXor,
          LogicalAnd, LogicalOr}) {
      table[assoc][assoc] = OperatorPriority::LeftFirst;
    }

    // Assignment is given right-to-left associativity in order to support
    // chained assignment.
    table[SimpleAssignment][SimpleAssignment] = OperatorPriority::RightFirst;

    // For other operators, there isn't an obvious answer and we require
    // explicit parentheses.
  }

  OperatorPriority table[NumPrecedenceLevels][NumPrecedenceLevels];
};
}  // namespace

auto PrecedenceGroup::ForLeading(TokenKind kind)
    -> llvm::Optional<PrecedenceGroup> {
  switch (kind) {
    case TokenKind::NotKeyword():
      return PrecedenceGroup(LogicalPrefix);

    case TokenKind::Minus():
    case TokenKind::MinusMinus():
    case TokenKind::PlusPlus():
      return PrecedenceGroup(NumericPrefix);

    case TokenKind::Tilde():
      return PrecedenceGroup(BitwisePrefix);

    default:
      return llvm::None;
  }
}

auto PrecedenceGroup::ForTrailing(TokenKind kind) -> llvm::Optional<Trailing> {
  switch (kind) {
    // Assignment operators.
    case TokenKind::Equal():
      return Trailing{.level = SimpleAssignment, .is_binary = true};
    case TokenKind::PlusEqual():
    case TokenKind::MinusEqual():
    case TokenKind::StarEqual():
    case TokenKind::SlashEqual():
    case TokenKind::PercentEqual():
    case TokenKind::AmpEqual():
    case TokenKind::PipeEqual():
    case TokenKind::GreaterGreaterEqual():
    case TokenKind::LessLessEqual():
      return Trailing{.level = CompoundAssignment, .is_binary = true};

    // Logical operators.
    case TokenKind::AndKeyword():
      return Trailing{.level = LogicalAnd, .is_binary = true};
    case TokenKind::OrKeyword():
      return Trailing{.level = LogicalOr, .is_binary = true};

    // Bitwise operators.
    case TokenKind::Amp():
      return Trailing{.level = BitwiseAnd, .is_binary = true};
    case TokenKind::Pipe():
      return Trailing{.level = BitwiseOr, .is_binary = true};
    case TokenKind::XorKeyword():
      return Trailing{.level = BitwiseXor, .is_binary = true};
    case TokenKind::GreaterGreater():
    case TokenKind::LessLess():
      return Trailing{.level = BitShift, .is_binary = true};

    // Relational operators.
    case TokenKind::EqualEqual():
    case TokenKind::ExclaimEqual():
    case TokenKind::Less():
    case TokenKind::LessEqual():
    case TokenKind::Greater():
    case TokenKind::GreaterEqual():
    case TokenKind::LessEqualGreater():
      return Trailing{.level = Relational, .is_binary = true};

    // Addative operators.
    case TokenKind::Plus():
    case TokenKind::Minus():
      return Trailing{.level = Additive, .is_binary = true};

    // Multiplicative operators.
    case TokenKind::Star():
    case TokenKind::Slash():
      return Trailing{.level = Multiplicative, .is_binary = true};
    case TokenKind::Percent():
      return Trailing{.level = Modulo, .is_binary = true};

    // Postfix operators.
    case TokenKind::MinusMinus():
    case TokenKind::PlusPlus():
      return Trailing{.level = NumericPostfix, .is_binary = false};

    // Prefix-only operators.
    case TokenKind::Tilde():
    case TokenKind::NotKeyword():
      break;

    // Symbolic tokens that might be operators eventually.
    case TokenKind::Backslash():
    case TokenKind::Caret():
    case TokenKind::CaretEqual():
    case TokenKind::Comma():
    case TokenKind::TildeEqual():
    case TokenKind::Exclaim():
    case TokenKind::LessGreater():
    case TokenKind::Question():
    case TokenKind::Colon():
      break;

    // Symbolic tokens that are intentionally not operators.
    case TokenKind::At():
    case TokenKind::LessMinus():
    case TokenKind::MinusGreater():
    case TokenKind::EqualGreater():
    case TokenKind::ColonEqual():
    case TokenKind::Period():
    case TokenKind::Semi():
      break;

    default:
      break;
  }

  return llvm::None;
}

auto PrecedenceGroup::GetPriority(PrecedenceGroup left, PrecedenceGroup right)
    -> OperatorPriority {
  static constexpr OperatorPriorityTable lookup;
  return lookup.table[left.level][right.level];
}

}  // namespace Carbon
