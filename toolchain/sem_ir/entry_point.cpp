// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/entry_point.h"

#include "llvm/ADT/StringRef.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

static constexpr llvm::StringLiteral EntryPointFunction = "Run";

auto IsEntryPoint(const File& file, FunctionId function_id) -> bool {
  const auto& function = file.functions().Get(function_id);

  return function.parent_scope_id == NameScopeId::Package &&
         file.package_id() == PackageNameId::None &&
         function.name_id.has_value() &&
         file.names().GetAsStringIfIdentifier(function.name_id) ==
             EntryPointFunction;
}

}  // namespace Carbon::SemIR
