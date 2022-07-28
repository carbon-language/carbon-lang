// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
#define CARBON_EXPLORER_INTERPRETER_BUILTINS_H_

#include <optional>

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

    // Arithmetic.
    Negate,
    AddWith,
    SubWith,
    MulWith,
    ModWith,

    Last = ModWith
  };
  // TODO: In C++20, replace with `using enum Builtin;`.
  static constexpr Builtin As = Builtin::As;
  static constexpr Builtin ImplicitAs = Builtin::ImplicitAs;
  static constexpr Builtin Negate = Builtin::Negate;
  static constexpr Builtin AddWith = Builtin::AddWith;
  static constexpr Builtin SubWith = Builtin::SubWith;
  static constexpr Builtin MulWith = Builtin::MulWith;
  static constexpr Builtin ModWith = Builtin::ModWith;


  // Register a declaration that might be a builtin.
  void Register(Nonnull<const Declaration*> decl);

  // Get a registered builtin.
  auto Get(SourceLocation source_loc, Builtin builtin) const
      -> ErrorOr<Nonnull<const Declaration*>>;

  // Get the source name of a builtin.
  static constexpr auto GetName(Builtin builtin) -> const char* {
    return BuiltinNames[static_cast<int>(builtin)];
  }

 private:
  static constexpr int NumBuiltins = static_cast<int>(Builtin::Last) + 1;
  static constexpr const char* BuiltinNames[NumBuiltins] = {
      "As", "ImplicitAs", "Negate", "AddWith", "SubWith", "MulWith","ModWith"};

  std::optional<Nonnull<const Declaration*>> builtins_[NumBuiltins] = {};
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
