// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/builtins.h"

#include "explorer/common/error_builders.h"

using llvm::dyn_cast;

namespace Carbon {

void Builtins::Register(Nonnull<const Declaration*> decl) {
  if (auto* interface = dyn_cast<InterfaceDeclaration>(decl)) {
    if (interface->name() == GetName(Builtin::ImplicitAs)) {
      builtins_[static_cast<int>(Builtin::ImplicitAs)] = interface;
    }
  }
}

auto Builtins::Get(SourceLocation source_loc, Builtin builtin) const
    -> ErrorOr<Nonnull<const Declaration*>> {
  std::optional<const Declaration*> result =
      builtins_[static_cast<int>(builtin)];
  if (!result.has_value()) {
    return CompilationError(source_loc)
           << "missing declaration for builtin `" << GetName(builtin) << "`";
  }
  return result.value();
}

}  // namespace Carbon
