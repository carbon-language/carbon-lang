// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/lower.h"

#include "toolchain/lower/file_context.h"

namespace Carbon::Lower {

auto LowerToLLVM(llvm::LLVMContext& llvm_context,
                 std::optional<llvm::ArrayRef<Parse::GetTreeAndSubtreesFn>>
                     tree_and_subtrees_getters_for_debug_info,
                 llvm::StringRef module_name, const SemIR::File& sem_ir,
                 clang::ASTUnit* cpp_ast, const SemIR::InstNamer* inst_namer,
                 llvm::raw_ostream* vlog_stream)
    -> std::unique_ptr<llvm::Module> {
  FileContext context(llvm_context, tree_and_subtrees_getters_for_debug_info,
                      module_name, sem_ir, cpp_ast, inst_namer, vlog_stream);
  return context.Run();
}

}  // namespace Carbon::Lower
