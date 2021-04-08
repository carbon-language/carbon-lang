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
struct PrecedenceComparisonTable {
  constexpr PrecedenceComparisonTable() : table{} {
    // Start with a list of <higher precedence>, <lower precedence>
    // relationships.
    MarkHigherThan({NumericPrefix, NumericPostfix}, {Modulo, Multiplicative});
    MarkHigherThan({Multiplicative}, {Additive});
    MarkHigherThan({BitwisePrefix}, {BitwiseAnd, BitwiseOr, BitwiseXor});
    MarkHigherThan({Modulo, Additive, BitwiseAnd, BitwiseOr, BitwiseXor},
                   {SimpleAssignment, CompoundAssignment, Relational});
    MarkHigherThan({Relational, LogicalPrefix}, {LogicalAnd, LogicalOr});

    // Compute the transitive closure of the above relationships.
    MakeTransitivelyClosed();

    // Make the relation symmetric.
    MakeSymmetric();
  }

  constexpr void MarkHigherThan(
      std::initializer_list<PrecedenceLevel> higher_group,
      std::initializer_list<PrecedenceLevel> lower_group) {
    for (auto higher : higher_group) {
      for (auto lower : lower_group) {
        table[higher][lower] = Precedence::Higher;
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
          if (table[a][b] == Precedence::Higher) {
            for (int8_t c = 0; c != NumPrecedenceLevels; ++c) {
              if (table[b][c] == Precedence::Higher &&
                  table[a][c] != Precedence::Higher) {
                table[a][c] = Precedence::Higher;
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
        if (table[a][b] == Precedence::Higher) {
          if (table[b][a] == Precedence::Higher) {
            throw "inconsistent lookup table entries";
          }
          table[b][a] = Precedence::Lower;
        }
      }
    }
  }

  Precedence table[NumPrecedenceLevels][NumPrecedenceLevels];
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

auto PrecedenceGroup::GetAssociativity() const -> Associativity {
  switch (static_cast<PrecedenceLevel>(level)) {
    // Prefix operators are modeled as having left-to-right associativity, even
    // though the question is not really applicable.
    case NumericPrefix:
    case BitwisePrefix:
    case LogicalPrefix:
      return Associativity::LeftToRight;

    // Postfix operators are modeled as having right-to-left associativity,
    // even though the question is not really applicable.
    case NumericPostfix:
      return Associativity::RightToLeft;

    // Traditionally-associative operators are given left-to-right
    // associativity.
    case Multiplicative:
    case Additive:
    case BitwiseAnd:
    case BitwiseOr:
    case BitwiseXor:
    case LogicalAnd:
    case LogicalOr:
      return Associativity::LeftToRight;

    // Assignment is given right-to-left associativity in order to support
    // chained assignment.
    case SimpleAssignment:
      return Associativity::RightToLeft;

    // If there isn't an obvious answer, we require explicit parentheses.
    case Modulo:
    case CompoundAssignment:
    case BitShift:
    case Relational:
      return Associativity::None;
  }
}

auto PrecedenceGroup::Compare(PrecedenceGroup a, PrecedenceGroup b)
    -> Precedence {
  static constexpr PrecedenceComparisonTable lookup;
  return lookup.table[a.level][b.level];
}

}  // namespace Carbon
