// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_H_

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// Use LowerToLLVM rather than calling this directly.
//
// This carries state for lowering. `Run()` should only be called once, and
// handles the main execution.
class Lowering {
 public:
  Lowering(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
           const SemanticsIR& semantics_ir);

  auto Run() -> std::unique_ptr<llvm::Module>;

 private:
  std::unique_ptr<llvm::Module> llvm_module_;
  const SemanticsIR* const semantics_ir_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_H_
