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

struct LowerToLLVMOptions {
  // Options must be set individually, not through initialization.
  explicit LowerToLLVMOptions() = default;

  // If set, enables LLVM IR verification.
  llvm::raw_ostream* llvm_verifier_stream = nullptr;

  // Whether to include debug info in lowered output.
  bool want_debug_info = false;

  // If set, enables verbose output.
  llvm::raw_ostream* vlog_stream = nullptr;

  // If set, LLVM IR will be dumped to this in textual form.
  llvm::raw_ostream* dump_stream = nullptr;
};

// Lowers SemIR to LLVM IR.
auto LowerToLLVM(
    llvm::LLVMContext& llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    const Parse::GetTreeAndSubtreesStore& tree_and_subtrees_getters,
    const SemIR::File& sem_ir, const LowerToLLVMOptions& options)
    -> std::unique_ptr<llvm::Module>;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_LOWER_H_
