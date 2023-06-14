// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_function_context.h"

#include "common/vlog.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

LoweringFunctionContext::LoweringFunctionContext(
    LoweringContext& lowering_context, llvm::Function* function)
    : lowering_context_(&lowering_context),
      function_(function),
      builder_(lowering_context.llvm_context()) {
  builder_.SetInsertPoint(
      llvm::BasicBlock::Create(llvm_context(), "entry", function_));
}

auto LoweringFunctionContext::GetLocalLoaded(SemanticsNodeId node_id)
    -> llvm::Value* {
  auto* value = GetLocal(node_id);
  if (llvm::isa<llvm::AllocaInst, llvm::GetElementPtrInst>(value)) {
    auto* load_type = GetType(semantics_ir().GetNode(node_id).type_id());
    return builder().CreateLoad(load_type, value);
  } else {
    // No load is needed.
    return value;
  }
}

}  // namespace Carbon
