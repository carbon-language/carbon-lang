// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_context.h"

#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node_kind.h"

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

auto LoweringContext::BuildLoweredNodeAsType(SemanticsNodeId node_id)
    -> llvm::Type* {
  switch (node_id.index) {
    case SemanticsBuiltinKind::EmptyStructType.AsInt():
    case SemanticsBuiltinKind::EmptyTuple.AsInt():
    case SemanticsBuiltinKind::EmptyTupleType.AsInt():
      // Represent empty data as 0-length arrays.
      // TODO: Work to remove EmptyTuple here.
      return llvm::ArrayType::get(builder_.getInt8Ty(), 0);
    case SemanticsBuiltinKind::FloatingPointType.AsInt():
      // TODO: Handle different sizes.
      return builder_.getDoubleTy();
    case SemanticsBuiltinKind::IntegerType.AsInt():
      // TODO: Handle different sizes.
      return builder_.getInt32Ty();
  }

  auto node = semantics_ir_->GetNode(node_id);
  switch (node.kind()) {
    case SemanticsNodeKind::StructType: {
      auto refs = semantics_ir_->GetNodeBlock(node.GetAsStructType().second);
      llvm::SmallVector<llvm::Type*> subtypes;
      subtypes.reserve(refs.size());
      for (auto ref_id : refs) {
        auto type_id = semantics_ir_->GetNode(ref_id).type_id();
        // TODO: Handle recursive types. The restriction for builtins prevents
        // recursion while still letting them cache.
        CARBON_CHECK(type_id.index < SemanticsBuiltinKind::ValidCount)
            << type_id;
        subtypes.push_back(GetLoweredNodeAsType(type_id));
      }
      auto* type = llvm::StructType::create(*llvm_context_, subtypes);
      lowered_nodes_[node_id.index] = type;
      return type;
    }
    default: {
      CARBON_FATAL() << "Cannot use node as type: " << node_id;
    }
  }
}

auto LoweringContext::GetLoweredNodeAsType(SemanticsNodeId node_id)
    -> llvm::Type* {
  if (lowered_nodes_[node_id.index]) {
    return lowered_nodes_[node_id.index].get<llvm::Type*>();
  }

  auto* type = BuildLoweredNodeAsType(node_id);
  lowered_nodes_[node_id.index] = type;
  return type;
}

}  // namespace Carbon
