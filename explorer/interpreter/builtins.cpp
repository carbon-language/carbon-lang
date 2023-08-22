// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/builtins.h"

#include "explorer/base/error_builders.h"

using llvm::dyn_cast;

namespace Carbon {

CARBON_DEFINE_ENUM_CLASS_NAMES(Builtin) = {
#define CARBON_BUILTIN(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "explorer/interpreter/builtins.def"
};

void Builtins::Register(Nonnull<const Declaration*> decl) {
  if (const auto* interface = dyn_cast<InterfaceDeclaration>(decl)) {
    if (interface->name().is_qualified()) {
      return;
    }

    static std::map<std::string, int, std::less<>>* builtin_indexes = [] {
      std::map<std::string, int, std::less<>> builtin_indexes;
      for (int index = 0; index < Builtin::NumBuiltins; ++index) {
        builtin_indexes.emplace(Builtin::FromInt(index).name(), index);
      }
      return new auto(std::move(builtin_indexes));
    }();

    auto it = builtin_indexes->find(interface->name().inner_name());
    if (it != builtin_indexes->end()) {
      builtins_[it->second] = interface;
    }
  }
}

auto Builtins::Get(SourceLocation source_loc, Builtin builtin) const
    -> ErrorOr<Nonnull<const Declaration*>> {
  std::optional<const Declaration*> result = builtins_[builtin.AsInt()];
  if (!result.has_value()) {
    return ProgramError(source_loc)
           << "missing declaration for builtin `" << builtin << "`";
  }
  return result.value();
}

}  // namespace Carbon
