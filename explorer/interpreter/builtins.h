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
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/value.h"

namespace Carbon {

class Builtins {
 public:
  explicit Builtins() {}

  enum class Builtin {
    // Conversions.
    As,
    ImplicitAs,

    // Comparison.
    EqWith,

    // Arithmetic.
    Negate,
    AddWith,
    SubWith,
    MulWith,
    ModWith,

    // Bitwise and shift.
    BitComplement,
    BitAndWith,
    BitOrWith,
    BitXorWith,
    LeftShiftWith,
    RightShiftWith,

    Last = RightShiftWith
  };
  // TODO: In C++20, replace with `using enum Builtin;`.
  static constexpr Builtin As = Builtin::As;
  static constexpr Builtin ImplicitAs = Builtin::ImplicitAs;
  static constexpr Builtin EqWith = Builtin::EqWith;
  static constexpr Builtin Negate = Builtin::Negate;
  static constexpr Builtin AddWith = Builtin::AddWith;
  static constexpr Builtin SubWith = Builtin::SubWith;
  static constexpr Builtin MulWith = Builtin::MulWith;
  static constexpr Builtin ModWith = Builtin::ModWith;
  static constexpr Builtin BitComplement = Builtin::BitComplement;
  static constexpr Builtin BitAndWith = Builtin::BitAndWith;
  static constexpr Builtin BitOrWith = Builtin::BitOrWith;
  static constexpr Builtin BitXorWith = Builtin::BitXorWith;
  static constexpr Builtin LeftShiftWith = Builtin::LeftShiftWith;
  static constexpr Builtin RightShiftWith = Builtin::RightShiftWith;

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
      "As",        "ImplicitAs", "EqWith",        "Negate",        "AddWith",
      "SubWith",   "MulWith",    "ModWith",       "BitComplement", "BitAndWith",
      "BitOrWith", "BitXorWith", "LeftShiftWith", "RightShiftWith"};

  std::optional<Nonnull<const Declaration*>> builtins_[NumBuiltins] = {};
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
