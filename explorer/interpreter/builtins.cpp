// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/builtins.h"

#include "explorer/common/error_builders.h"

using llvm::dyn_cast;

namespace Carbon {

void Builtins::Register(Nonnull<const Declaration*> decl) {
  if (const auto* interface = dyn_cast<InterfaceDeclaration>(decl)) {
    static std::map<std::string, int>* builtin_indexes = [] {
      std::map<std::string, int> builtin_indexes;
      for (int index = 0; index <= static_cast<int>(Builtin::Last); ++index) {
        builtin_indexes.emplace(BuiltinNames[index], index);
      }
      return new auto(std::move(builtin_indexes));
    }();

    auto it = builtin_indexes->find(interface->name());
    if (it != builtin_indexes->end()) {
      builtins_[it->second] = interface;
    }
  }
}

auto Builtins::Get(SourceLocation source_loc, Builtin builtin) const
    -> ErrorOr<Nonnull<const Declaration*>> {
  std::optional<const Declaration*> result =
      builtins_[static_cast<int>(builtin)];
  if (!result.has_value()) {
    return ProgramError(source_loc)
           << "missing declaration for builtin `" << GetName(builtin) << "`";
  }
  return result.value();
}

Builtins::Builtin Builtins::BuiltinInterfaceForAssignOperator(
    AssignOperator op) {
  switch (op) {
    case AssignOperator::Plain:
      return Builtin::AssignFrom;
    case AssignOperator::Add:
      return Builtin::AssignAddWith;
    case AssignOperator::Sub:
      return Builtin::AssignSubWith;
    case AssignOperator::Mul:
      return Builtin::AssignMulWith;
    case AssignOperator::Div:
      return Builtin::AssignDivWith;
    case AssignOperator::Mod:
      return Builtin::AssignModWith;
    case AssignOperator::And:
      return Builtin::AssignBitAndWith;
    case AssignOperator::Or:
      return Builtin::AssignBitOrWith;
    case AssignOperator::Xor:
      return Builtin::AssignBitXorWith;
    case AssignOperator::ShiftLeft:
      return Builtin::AssignLeftShiftWith;
    case AssignOperator::ShiftRight:
      return Builtin::AssignRightShiftWith;
  }
}

}  // namespace Carbon
