// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/entry_point.h"

#include "llvm/ADT/StringRef.h"

namespace Carbon::SemIR {

static constexpr llvm::StringLiteral EntryPointFunction = "Run";

auto IsEntryPoint(const SemIR::File& file, SemIR::FunctionId function_id)
    -> bool {
  // TODO: Check if `file` is in the `Main` package.
  const auto& function = file.functions().Get(function_id);
  // TODO: Check if `function` is in a namespace.
  return function.name_id.is_valid() &&
         file.names().GetAsStringIfIdentifier(function.name_id) ==
             EntryPointFunction;
}

}  // namespace Carbon::SemIR
