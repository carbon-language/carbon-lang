// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering.h"

namespace Carbon {

Lowering::Lowering(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                   const SemanticsIR& semantics_ir)
    : llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      semantics_ir_(&semantics_ir) {}

auto Lowering::Run() -> std::unique_ptr<llvm::Module> {
  CARBON_CHECK(llvm_module_) << "Run can only be called once.";

  return std::move(llvm_module_);
}

}  // namespace Carbon
