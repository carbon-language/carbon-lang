// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_LOWER_H_
#define CARBON_TOOLCHAIN_LOWER_LOWER_H_

#include "common/ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// Lowers Semantics IR to LLVM IR.
class Lower {
 public:
  static auto Make(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                   const SemanticsIR& semantics_ir)
      -> std::unique_ptr<llvm::Module>;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWER_LOWER_H_
