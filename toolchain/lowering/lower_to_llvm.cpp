// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lower_to_llvm.h"

namespace Carbon {

auto LowerToLLVM(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                 const SemanticsIR& /*semantics_ir*/)
    -> std::unique_ptr<llvm::Module> {
  auto result = std::make_unique<llvm::Module>(module_name, llvm_context);
  return result;
}

}  // namespace Carbon
