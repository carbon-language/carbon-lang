// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/precedence.h"

#include <utility>

#include "common/check.h"

namespace Carbon {

namespace {
enum PrecedenceLevel : int8_t {
  // Sentinel representing the absence of any operator.
  Highest,
  // Terms.
  TermPrefix,
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
  // Type formation.
  TypePostfix,
  // Sentinel representing a type context.
  Type,
  // Logical.
  LogicalPrefix,
  Relational,
  LogicalAnd,
  LogicalOr,
  // Assignment.
  SimpleAssignment,
  CompoundAssignment,
  // Sentinel representing a context in which any operator can appear.
  Lowest,
};
constexpr int8_t NumPrecedenceLevels = Lowest + 1;

// A precomputed lookup table determining the relative precedence of two
// precedence groups.
struct OperatorPriorityTable {
  constexpr OperatorPriorityTable() : table() {
    // Start with a list of <higher precedence>, <lower precedence>
    // relationships.
    MarkHigherThan({Highest}, {TermPrefix});
    MarkHigherThan({TermPrefix}, {NumericPrefix, BitwisePrefix, LogicalPrefix,
                                  NumericPostfix, TypePostfix});
    MarkHigherThan({NumericPrefix, NumericPostfix},
                   {Modulo, Multiplicative, BitShift});
    MarkHigherThan({Multiplicative}, {Additive});
    MarkHigherThan({BitwisePrefix},
                   {BitwiseAnd, BitwiseOr, BitwiseXor, BitShift});
    MarkHigherThan({TypePostfix}, {Type});
    MarkHigherThan(
        {Modulo, Additive, BitwiseAnd, BitwiseOr, BitwiseXor, BitShift, Type},
        {SimpleAssignment, CompoundAssignment, Relational});
    MarkHigherThan({Relational, LogicalPrefix}, {LogicalAnd, LogicalOr});
    MarkHigherThan(
        {SimpleAssignment, CompoundAssignment, LogicalAnd, LogicalOr},
        {Lowest});

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

    ConsistencyCheck();
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
      // NOLINTNEXTLINE(modernize-loop-convert)
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
          CARBON_CHECK(table[b][a] != OperatorPriority::LeftFirst)
              << "inconsistent lookup table entries";
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
         {TermPrefix, NumericPrefix, BitwisePrefix, LogicalPrefix}) {
      table[prefix][prefix] = OperatorPriority::RightFirst;
    }

    // Postfix operators are symmetric with prefix operators.
    for (PrecedenceLevel postfix : {NumericPostfix, TypePostfix}) {
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

  constexpr void ConsistencyCheck() {
    for (int8_t level = 0; level != NumPrecedenceLevels; ++level) {
      if (level != Highest) {
        CARBON_CHECK(table[Highest][level] == OperatorPriority::LeftFirst &&
                     table[level][Highest] == OperatorPriority::RightFirst)
            << "Highest is not highest priority";
      }
      if (level != Lowest) {
        CARBON_CHECK(table[Lowest][level] == OperatorPriority::RightFirst &&
                     table[level][Lowest] == OperatorPriority::LeftFirst)
            << "Lowest is not lowest priority";
      }
    }
  }

  OperatorPriority table[NumPrecedenceLevels][NumPrecedenceLevels];
};
}  // namespace

auto PrecedenceGroup::ForPostfixExpression() -> PrecedenceGroup {
  return PrecedenceGroup(Highest);
}

auto PrecedenceGroup::ForTopLevelExpression() -> PrecedenceGroup {
  return PrecedenceGroup(Lowest);
}

auto PrecedenceGroup::ForType() -> PrecedenceGroup {
  return PrecedenceGroup(Type);
}

auto PrecedenceGroup::ForLeading(TokenKind kind)
    -> std::optional<PrecedenceGroup> {
  switch (kind) {
    case TokenKind::Star:
      return PrecedenceGroup(TermPrefix);

    case TokenKind::Not:
      return PrecedenceGroup(LogicalPrefix);

    case TokenKind::Minus:
    case TokenKind::MinusMinus:
    case TokenKind::PlusPlus:
      return PrecedenceGroup(NumericPrefix);

    case TokenKind::Tilde:
      return PrecedenceGroup(BitwisePrefix);

    default:
      return std::nullopt;
  }
}

auto PrecedenceGroup::ForTrailing(TokenKind kind, bool infix)
    -> std::optional<Trailing> {
  switch (kind) {
    // Assignment operators.
    case TokenKind::Equal:
      return Trailing{.level = SimpleAssignment, .is_binary = true};
    case TokenKind::PlusEqual:
    case TokenKind::MinusEqual:
    case TokenKind::StarEqual:
    case TokenKind::SlashEqual:
    case TokenKind::PercentEqual:
    case TokenKind::AmpEqual:
    case TokenKind::PipeEqual:
    case TokenKind::GreaterGreaterEqual:
    case TokenKind::LessLessEqual:
      return Trailing{.level = CompoundAssignment, .is_binary = true};

    // Logical operators.
    case TokenKind::And:
      return Trailing{.level = LogicalAnd, .is_binary = true};
    case TokenKind::Or:
      return Trailing{.level = LogicalOr, .is_binary = true};

    // Bitwise operators.
    case TokenKind::Amp:
      return Trailing{.level = BitwiseAnd, .is_binary = true};
    case TokenKind::Pipe:
      return Trailing{.level = BitwiseOr, .is_binary = true};
    case TokenKind::Xor:
      return Trailing{.level = BitwiseXor, .is_binary = true};
    case TokenKind::GreaterGreater:
    case TokenKind::LessLess:
      return Trailing{.level = BitShift, .is_binary = true};

    // Relational operators.
    case TokenKind::EqualEqual:
    case TokenKind::ExclaimEqual:
    case TokenKind::Less:
    case TokenKind::LessEqual:
    case TokenKind::Greater:
    case TokenKind::GreaterEqual:
    case TokenKind::LessEqualGreater:
      return Trailing{.level = Relational, .is_binary = true};

    // Additive operators.
    case TokenKind::Plus:
    case TokenKind::Minus:
      return Trailing{.level = Additive, .is_binary = true};

    // Multiplicative operators.
    case TokenKind::Slash:
      return Trailing{.level = Multiplicative, .is_binary = true};
    case TokenKind::Percent:
      return Trailing{.level = Modulo, .is_binary = true};

    // `*` could be multiplication or pointer type formation.
    case TokenKind::Star:
      return infix ? Trailing{.level = Multiplicative, .is_binary = true}
                   : Trailing{.level = TypePostfix, .is_binary = false};

    // Postfix operators.
    case TokenKind::MinusMinus:
    case TokenKind::PlusPlus:
      return Trailing{.level = NumericPostfix, .is_binary = false};

    // Prefix-only operators.
    case TokenKind::Tilde:
    case TokenKind::Not:
      break;

    // Symbolic tokens that might be operators eventually.
    case TokenKind::Backslash:
    case TokenKind::Caret:
    case TokenKind::CaretEqual:
    case TokenKind::Comma:
    case TokenKind::TildeEqual:
    case TokenKind::Exclaim:
    case TokenKind::LessGreater:
    case TokenKind::Question:
    case TokenKind::Colon:
      break;

    // Symbolic tokens that are intentionally not operators.
    case TokenKind::At:
    case TokenKind::LessMinus:
    case TokenKind::MinusGreater:
    case TokenKind::EqualGreater:
    case TokenKind::ColonEqual:
    case TokenKind::Period:
    case TokenKind::Semi:
      break;

    default:
      break;
  }

  return std::nullopt;
}

auto PrecedenceGroup::GetPriority(PrecedenceGroup left, PrecedenceGroup right)
    -> OperatorPriority {
  static constexpr OperatorPriorityTable Lookup;
  return Lookup.table[left.level_][right.level_];
}

}  // namespace Carbon
