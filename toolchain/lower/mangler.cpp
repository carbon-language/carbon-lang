// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/mangler.h"

#include "toolchain/sem_ir/entry_point.h"

namespace Carbon::Lower {

auto Mangler::Mangle(SemIR::FunctionId function_id) -> std::string {
  const auto& function = sem_ir_.functions().Get(function_id);
  if (SemIR::IsEntryPoint(sem_ir_, function_id)) {
    // TODO: Add an implicit `return 0` if `Run` doesn't return `i32`.
    return "main";
  }
  // TODO: Decide on a name mangling scheme.
  auto name = sem_ir_.names().GetAsStringIfIdentifier(function.name_id);
  CARBON_CHECK(name) << "Unexpected special name for function: "
                     << function.name_id;
  SemIR::NameScopeId parent_scope_id = function.parent_scope_id;
  std::string result = "_C" + name->str();
  while (parent_scope_id.is_valid()) {
    const auto& parent = sem_ir_.name_scopes().Get(parent_scope_id);
    if (parent.name_id == SemIR::NameId::PackageNamespace) {
      break;
    }
    auto name = sem_ir_.names().GetAsStringIfIdentifier(parent.name_id);
    if (!name) {
      break;
    }
    CARBON_CHECK(name) << "Unexpected special name for function scope: "
                       << function.name_id;
    result += '.';
    result += *name;
    parent_scope_id = parent.parent_scope_id;
  }
  return result;
}

}  // namespace Carbon::Lower
