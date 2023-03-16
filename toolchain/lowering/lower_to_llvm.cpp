// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lower_to_llvm.h"

#include "toolchain/lowering/lowering.h"

namespace Carbon {

auto LowerToLLVM(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                 const SemanticsIR& semantics_ir)
    -> std::unique_ptr<llvm::Module> {
  Lowering lowering(llvm_context, module_name, semantics_ir);
  return lowering.Run();
}

}  // namespace Carbon
