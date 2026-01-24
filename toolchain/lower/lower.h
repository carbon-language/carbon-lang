// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_LOWER_H_
#define CARBON_TOOLCHAIN_LOWER_LOWER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/lower/options.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

// Lowers SemIR to LLVM IR.
auto LowerToLLVM(
    llvm::LLVMContext& llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    const Parse::GetTreeAndSubtreesStore& tree_and_subtrees_getters,
    const SemIR::File& sem_ir, int total_ir_count,
    const LowerToLLVMOptions& options) -> std::unique_ptr<llvm::Module>;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_LOWER_H_
