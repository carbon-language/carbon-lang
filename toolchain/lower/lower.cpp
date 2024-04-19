// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/lower.h"

#include "toolchain/lower/file_context.h"

namespace Carbon::Lower {

auto LowerToLLVM(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                 const SemIR::File& sem_ir, const SemIR::InstNamer* inst_namer,
                 llvm::raw_ostream* vlog_stream)
    -> std::unique_ptr<llvm::Module> {
  FileContext context(llvm_context, module_name, sem_ir, inst_namer,
                      vlog_stream);
  return context.Run();
}

}  // namespace Carbon::Lower
