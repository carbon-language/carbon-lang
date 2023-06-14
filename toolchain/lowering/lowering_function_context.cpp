// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_function_context.h"

#include "common/vlog.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

LoweringFunctionContext::LoweringFunctionContext(
    LoweringContext& lowering_context, llvm::Function* function,
    llvm::raw_ostream* vlog_stream)
    : lowering_context_(&lowering_context),
      function_(function),
      builder_(lowering_context.llvm_context()),
      vlog_stream_(vlog_stream) {
  builder_.SetInsertPoint(
      llvm::BasicBlock::Create(llvm_context(), "entry", function));
}

auto LoweringFunctionContext::BuildFunctionDefinition(
    const SemanticsFunction& function) -> void {
  // Add parameters to locals.
  auto param_refs = semantics_ir().GetNodeBlock(function.param_refs_id);
  for (int i = 0; i < static_cast<int>(param_refs.size()); ++i) {
    auto param_storage =
        semantics_ir().GetNode(param_refs[i]).GetAsBindName().second;
    CARBON_CHECK(locals_.insert({param_storage, function_->getArg(i)}).second)
        << "Duplicate param: " << param_refs[i];
  }

  CARBON_VLOG() << "Lowering " << function.body_id << "\n";
  for (const auto& node_id : semantics_ir().GetNodeBlock(function.body_id)) {
    auto node = semantics_ir().GetNode(node_id);
    CARBON_VLOG() << "Lowering " << node_id << ": " << node << "\n";
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)        \
  case SemanticsNodeKind::Name:                 \
    LoweringHandle##Name(*this, node_id, node); \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
  }
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
