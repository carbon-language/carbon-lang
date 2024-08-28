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
  return "_C" + name->str();
}

}  // namespace Carbon::Lower
