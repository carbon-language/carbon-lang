// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/lower.h"

#include <memory>
#include <optional>

#include "common/vlog.h"
#include "llvm/IR/Verifier.h"
#include "toolchain/lower/context.h"
#include "toolchain/lower/file_context.h"

namespace Carbon::Lower {

auto LowerToLLVM(
    llvm::LLVMContext& llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    const Parse::GetTreeAndSubtreesStore& tree_and_subtrees_getters,
    const SemIR::File& sem_ir, int total_ir_count,
    const LowerToLLVMOptions& options) -> std::unique_ptr<llvm::Module> {
  Context context(&llvm_context, std::move(fs), options.want_debug_info,
                  &tree_and_subtrees_getters, sem_ir.filename(), total_ir_count,
                  options.vlog_stream);

  // TODO: Consider disabling instruction naming by default if we're not
  // producing textual LLVM IR.
  SemIR::InstNamer inst_namer(&sem_ir, total_ir_count);
  context.GetFileContext(&sem_ir, &inst_namer).LowerDefinitions();

  std::unique_ptr<llvm::Module> module = std::move(context).Finalize();

  if (options.vlog_stream) {
    CARBON_VLOG_TO(options.vlog_stream, "*** llvm::Module ***\n");
    module->print(*options.vlog_stream, /*AAW=*/nullptr,
                  /*ShouldPreserveUseListOrder=*/false,
                  /*IsForDebug=*/true);
  }
  if (options.dump_stream) {
    module->print(*options.dump_stream, /*AAW=*/nullptr,
                  /*ShouldPreserveUseListOrder=*/true);
  }

  if (options.llvm_verifier_stream) {
    CARBON_CHECK(!llvm::verifyModule(*module, options.llvm_verifier_stream));
  }

  return module;
}

}  // namespace Carbon::Lower
