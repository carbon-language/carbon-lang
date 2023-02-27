// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_H_

#include "llvm/IR/IRBuilder.h"
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
  explicit Lowering(llvm::LLVMContext& llvm_context,
                    llvm::StringRef module_name,
                    const SemanticsIR& semantics_ir);

  auto Run() -> std::unique_ptr<llvm::Module>;

 private:
  // Declare handlers for each SemanticsIR node.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  auto Handle##Name##Node(SemanticsNode node)->void;
#include "toolchain/semantics/semantics_node_kind.def"

  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;
  llvm::IRBuilder<> builder_;
  const SemanticsIR* const semantics_ir_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_H_
