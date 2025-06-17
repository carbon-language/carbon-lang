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
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs, bool want_debug_info,
    llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
    llvm::StringRef module_name, const SemIR::File& sem_ir,
    const SemIR::InstNamer* inst_namer, llvm::raw_ostream* vlog_stream)
    -> std::unique_ptr<llvm::Module> {
  Context context(llvm_context, std::move(fs), want_debug_info,
                  tree_and_subtrees_getters, module_name, vlog_stream);
  context.GetFileContext(&sem_ir, inst_namer).LowerDefinitions();
  return std::move(context).Finalize();
}

}  // namespace Carbon::Lower
