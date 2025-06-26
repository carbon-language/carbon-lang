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

struct LowerToLLVMParams {
  // Must be non-null.
  llvm::LLVMContext* llvm_context;

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs;

  // Optionally provided to enable verification.
  llvm::raw_ostream* llvm_verifier_stream;

  bool want_debug_info;
  llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters;
  llvm::StringRef module_name;

  // Must be non-null.
  const SemIR::File* sem_ir;

  // Must be non-null.
  const SemIR::InstNamer* inst_namer;

  // Optionally provided to enable VLOG output.
  llvm::raw_ostream* vlog_stream = nullptr;
};

// Lowers SemIR to LLVM IR.
auto LowerToLLVM(LowerToLLVMParams params) -> std::unique_ptr<llvm::Module>;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_LOWER_H_
