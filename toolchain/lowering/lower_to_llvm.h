// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWER_TO_LLVM_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWER_TO_LLVM_H_

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// Lowers Semantics IR to LLVM IR.
auto LowerToLLVM(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                 const SemanticsIR& semantics_ir)
    -> std::unique_ptr<llvm::Module>;

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWER_TO_LLVM_H_
