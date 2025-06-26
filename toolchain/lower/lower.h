// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_LOWER_H_
#define CARBON_TOOLCHAIN_LOWER_LOWER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

// Lowers SemIR to LLVM IR.
//
// `llvm_verifier_stream` should be non-null when verification is desired.
// TODO: Switch to a struct for parameters.
auto LowerToLLVM(
    llvm::LLVMContext& llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::raw_ostream* llvm_verifier_stream, bool want_debug_info,
    llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
    llvm::StringRef module_name, const SemIR::File& sem_ir,
    const SemIR::InstNamer* inst_namer, llvm::raw_ostream* vlog_stream)
    -> std::unique_ptr<llvm::Module>;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_LOWER_H_
