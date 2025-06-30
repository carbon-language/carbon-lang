// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/lower.h"

#include <memory>
#include <optional>

#include "toolchain/lower/context.h"
#include "toolchain/lower/file_context.h"

namespace Carbon::Lower {

auto LowerToLLVM(
    llvm::LLVMContext& llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
    const SemIR::File& sem_ir, const LowerToLLVMOptions& options)
    -> std::unique_ptr<llvm::Module> {
  Context context(llvm_context, std::move(fs), options.want_debug_info,
                  tree_and_subtrees_getters, sem_ir.filename(),
                  options.vlog_stream);

  // TODO: Consider disabling instruction naming by default if we're not
  // producing textual LLVM IR.
  SemIR::InstNamer inst_namer(&sem_ir);
  context.GetFileContext(&sem_ir, &inst_namer).LowerDefinitions();

  return std::move(context).Finalize(options.llvm_verifier_stream);
}

}  // namespace Carbon::Lower
