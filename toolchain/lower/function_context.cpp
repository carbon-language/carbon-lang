// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {

FunctionContext::FunctionContext(FileContext& file_context,
                                 llvm::Function* function)
    : file_context_(&file_context),
      function_(function),
      builder_(file_context.llvm_context()) {}

auto FunctionContext::GetBlock(SemIR::NodeBlockId block_id)
    -> llvm::BasicBlock* {
  llvm::BasicBlock*& entry = blocks_[block_id];
  if (!entry) {
    entry = llvm::BasicBlock::Create(llvm_context(), "", function_);
  }
  return entry;
}

auto FunctionContext::TryToReuseBlock(SemIR::NodeBlockId block_id,
                                      llvm::BasicBlock* block) -> bool {
  if (!blocks_.insert({block_id, block}).second) {
    return false;
  }
  if (block == synthetic_block_) {
    synthetic_block_ = nullptr;
  }
  return true;
}

auto FunctionContext::GetBlockArg(SemIR::NodeBlockId block_id,
                                  SemIR::TypeId type_id) -> llvm::PHINode* {
  llvm::BasicBlock* block = GetBlock(block_id);

  // Find the existing phi, if any.
  auto phis = block->phis();
  if (!phis.empty()) {
    CARBON_CHECK(std::next(phis.begin()) == phis.end())
        << "Expected at most one phi, found "
        << std::distance(phis.begin(), phis.end());
    return &*phis.begin();
  }

  // The number of predecessor slots to reserve.
  static constexpr unsigned NumReservedPredecessors = 2;
  auto* phi = llvm::PHINode::Create(GetType(type_id), NumReservedPredecessors);
  phi->insertInto(block, block->begin());
  return phi;
}

auto FunctionContext::CreateSyntheticBlock() -> llvm::BasicBlock* {
  synthetic_block_ = llvm::BasicBlock::Create(llvm_context(), "", function_);
  return synthetic_block_;
}

auto FunctionContext::FinishInitialization(SemIR::TypeId type_id,
                                           SemIR::NodeId dest_id,
                                           SemIR::NodeId source_id) -> void {
  switch (SemIR::GetInitializingRepresentation(semantics_ir(), type_id).kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      break;
    case SemIR::InitializingRepresentation::ByCopy:
      CopyValue(type_id, source_id, dest_id);
      break;
  }
}

auto FunctionContext::CopyValue(SemIR::TypeId type_id, SemIR::NodeId source_id,
                                SemIR::NodeId dest_id) -> void {
  switch (auto rep = SemIR::GetValueRepresentation(semantics_ir(), type_id);
          rep.kind) {
    case SemIR::ValueRepresentation::None:
      break;
    case SemIR::ValueRepresentation::Copy:
      builder().CreateStore(GetLocalLoaded(source_id), GetLocal(dest_id));
      break;
    case SemIR::ValueRepresentation::Pointer: {
      const auto& layout = llvm_module().getDataLayout();
      auto* type = GetType(type_id);
      // TODO: Compute known alignment of the source and destination, which may
      // be greater than the alignment computed by LLVM.
      auto align = layout.getABITypeAlign(type);

      // TODO: Attach !tbaa.struct metadata indicating which portions of the
      // type we actually need to copy and which are padding.
      builder().CreateMemCpy(GetLocal(dest_id), align, GetLocal(source_id),
                             align, layout.getTypeAllocSize(type));
      break;
    }
    case SemIR::ValueRepresentation::Custom:
      CARBON_FATAL() << "TODO: Add support for CopyValue with custom value rep";
  }
}

auto FunctionContext::GetLocalLoaded(SemIR::NodeId node_id) -> llvm::Value* {
  auto* value = GetLocal(node_id);
  if (llvm::isa<llvm::AllocaInst, llvm::GetElementPtrInst>(value)) {
    auto* load_type = GetType(semantics_ir().GetNode(node_id).type_id());
    return builder().CreateLoad(load_type, value);
  } else {
    // No load is needed.
    return value;
  }
}

}  // namespace Carbon::Lower
