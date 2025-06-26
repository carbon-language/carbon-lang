// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/lower.h"

#include <memory>
#include <optional>

#include "toolchain/lower/context.h"
#include "toolchain/lower/file_context.h"

namespace Carbon::Lower {

auto LowerToLLVM(LowerToLLVMParams params) -> std::unique_ptr<llvm::Module> {
  Context context(*params.llvm_context, std::move(params.fs),
                  params.want_debug_info, params.tree_and_subtrees_getters,
                  params.module_name, params.vlog_stream);
  context.GetFileContext(params.sem_ir, params.inst_namer).LowerDefinitions();
  return std::move(context).Finalize(params.llvm_verifier_stream);
}

}  // namespace Carbon::Lower
