// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
#define CARBON_EXPLORER_INTERPRETER_BUILTINS_H_

#include <array>
#include <optional>
#include <string_view>

#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/value.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"

namespace Carbon {

class Builtins {
 public:
  explicit Builtins() = default;

  enum class Builtin {
    // Conversions.
    As,
    ImplicitAs,

    // Comparison.
    EqWith,
    LessWith,
    LessEqWith,
    GreaterWith,
    GreaterEqWith,
    CompareWith,

    // Arithmetic.
    Negate,
    AddWith,
    SubWith,
    MulWith,
    DivWith,
    ModWith,

    // Bitwise and shift.
    BitComplement,
    BitAndWith,
    BitOrWith,
    BitXorWith,
    LeftShiftWith,
    RightShiftWith,

    // Simple assignment.
    AssignWith,

    // Compound assignment.
    AddAssignWith,
    SubAssignWith,
    MulAssignWith,
    DivAssignWith,
    ModAssignWith,
    BitAndAssignWith,
    BitOrAssignWith,
    BitXorAssignWith,
    LeftShiftAssignWith,
    RightShiftAssignWith,

    // Increment and decrement.
    Inc,
    Dec,

    Last = Dec
  };
  // TODO: In C++20, replace with `using enum Builtin;`.
  static constexpr Builtin As = Builtin::As;
  static constexpr Builtin ImplicitAs = Builtin::ImplicitAs;
  static constexpr Builtin EqWith = Builtin::EqWith;
  static constexpr Builtin LessWith = Builtin::LessWith;
  static constexpr Builtin LessEqWith = Builtin::LessEqWith;
  static constexpr Builtin GreaterWith = Builtin::GreaterWith;
  static constexpr Builtin GreaterEqWith = Builtin::GreaterEqWith;
  static constexpr Builtin CompareWith = Builtin::CompareWith;
  static constexpr Builtin Negate = Builtin::Negate;
  static constexpr Builtin AddWith = Builtin::AddWith;
  static constexpr Builtin SubWith = Builtin::SubWith;
  static constexpr Builtin MulWith = Builtin::MulWith;
  static constexpr Builtin DivWith = Builtin::DivWith;
  static constexpr Builtin ModWith = Builtin::ModWith;
  static constexpr Builtin BitComplement = Builtin::BitComplement;
  static constexpr Builtin BitAndWith = Builtin::BitAndWith;
  static constexpr Builtin BitOrWith = Builtin::BitOrWith;
  static constexpr Builtin BitXorWith = Builtin::BitXorWith;
  static constexpr Builtin LeftShiftWith = Builtin::LeftShiftWith;
  static constexpr Builtin RightShiftWith = Builtin::RightShiftWith;
  static constexpr Builtin AssignWith = Builtin::AssignWith;
  static constexpr Builtin AddAssignWith = Builtin::AddAssignWith;
  static constexpr Builtin SubAssignWith = Builtin::SubAssignWith;
  static constexpr Builtin MulAssignWith = Builtin::MulAssignWith;
  static constexpr Builtin DivAssignWith = Builtin::DivAssignWith;
  static constexpr Builtin ModAssignWith = Builtin::ModAssignWith;
  static constexpr Builtin BitAndAssignWith = Builtin::BitAndAssignWith;
  static constexpr Builtin BitOrAssignWith = Builtin::BitOrAssignWith;
  static constexpr Builtin BitXorAssignWith = Builtin::BitXorAssignWith;
  static constexpr Builtin LeftShiftAssignWith = Builtin::LeftShiftAssignWith;
  static constexpr Builtin RightShiftAssignWith = Builtin::RightShiftAssignWith;
  static constexpr Builtin Inc = Builtin::Inc;
  static constexpr Builtin Dec = Builtin::Dec;

  // Register a declaration that might be a builtin.
  void Register(Nonnull<const Declaration*> decl);

  // Get a registered builtin.
  auto Get(SourceLocation source_loc, Builtin builtin) const
      -> ErrorOr<Nonnull<const Declaration*>>;

  // Get the source name of a builtin.
  static constexpr auto GetName(Builtin builtin) -> std::string_view {
    return BuiltinNames[static_cast<int>(builtin)];
  }

 private:
  static constexpr int NumBuiltins = static_cast<int>(Builtin::Last) + 1;
  static constexpr const char* BuiltinNames[NumBuiltins] = {
      "As",
      "ImplicitAs",
      "EqWith",
      "LessWith",
      "LessEqWith",
      "GreaterWith",
      "GreaterEqWith",
      "CompareWith",
      "Negate",
      "AddWith",
      "SubWith",
      "MulWith",
      "DivWith",
      "ModWith",
      "BitComplement",
      "BitAndWith",
      "BitOrWith",
      "BitXorWith",
      "LeftShiftWith",
      "RightShiftWith",
      "AssignWith",
      "AddAssignWith",
      "SubAssignWith",
      "MulAssignWith",
      "DivAssignWith",
      "ModAssignWith",
      "BitAndAssignWith",
      "BitOrAssignWith",
      "BitXorAssignWith",
      "LeftShiftAssignWith",
      "RightShiftAssignWith",
      "Inc",
      "Dec"};

  std::optional<Nonnull<const Declaration*>> builtins_[NumBuiltins] = {};
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
