// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_context.h"

#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

LoweringContext::LoweringContext(llvm::LLVMContext& llvm_context,
                                 llvm::StringRef module_name,
                                 const SemanticsIR& semantics_ir)
    : llvm_context_(&llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      builder_(llvm_context),
      semantics_ir_(&semantics_ir),
      lowered_nodes_(semantics_ir_->nodes_size(), nullptr) {
  CARBON_CHECK(!semantics_ir.has_errors())
      << "Generating LLVM IR from invalid SemanticsIR is unsupported.";
}

auto LoweringContext::Run() -> std::unique_ptr<llvm::Module> {
  CARBON_CHECK(llvm_module_) << "Run can only be called once.";

  LowerBlock(semantics_ir_->top_node_block_id());

  while (!todo_blocks_.empty()) {
    auto [llvm_block, block_id] = todo_blocks_.pop_back_val();
    builder_.SetInsertPoint(llvm_block);
    LowerBlock(block_id);
  }

  return std::move(llvm_module_);
}

auto LoweringContext::LowerBlock(SemanticsNodeBlockId block_id) -> void {
  for (const auto& node_id : semantics_ir_->GetNodeBlock(block_id)) {
    auto node = semantics_ir_->GetNode(node_id);
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)        \
  case SemanticsNodeKind::Name:                 \
    LoweringHandle##Name(*this, node_id, node); \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
  }
}

auto LoweringContext::LowerNodeToType(SemanticsNodeId node_id) -> llvm::Type* {
  CARBON_CHECK(node_id.is_valid());
  switch (node_id.index) {
    case SemanticsBuiltinKind::EmptyTuple.AsInt():
      // TODO: Should probably switch this to an actual empty tuple in the
      // future, but it's implemented as void for now.
      return builder_.getVoidTy();
    case SemanticsBuiltinKind::IntegerType.AsInt():
      // TODO: Handle different sizes.
      return builder_.getInt32Ty();
    default:
      CARBON_FATAL() << "Cannot use node as type: " << node_id;
  }
}

}  // namespace Carbon
