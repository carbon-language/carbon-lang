// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/lower.h"

namespace Carbon {

auto LowerResult::Print(llvm::raw_ostream& output) const -> void {
  output << "TODO: Print IR";
}

auto Lower::Make(llvm::LLVMContext& llvm_context,
                 const SemanticsIR& /*semantics_ir*/) -> LowerResult {
  LowerResult result(llvm_context, "todo: module_id");
  return result;
}

}  // namespace Carbon
