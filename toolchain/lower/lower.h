// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_LOWER_H_
#define CARBON_TOOLCHAIN_LOWER_LOWER_H_

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {

// Lowers SemIR to LLVM IR.
auto LowerToLLVM(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                 const SemIR::File& sem_ir, llvm::raw_ostream* vlog_stream)
    -> std::unique_ptr<llvm::Module>;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_LOWER_H_
