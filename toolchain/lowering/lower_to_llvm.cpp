// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lower_to_llvm.h"

#include "toolchain/lowering/lowering_context.h"

namespace Carbon {

auto LowerToLLVM(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                 const SemanticsIR& semantics_ir)
    -> std::unique_ptr<llvm::Module> {
  LoweringContext context(llvm_context, module_name, semantics_ir);
  return context.Run();
}

}  // namespace Carbon
