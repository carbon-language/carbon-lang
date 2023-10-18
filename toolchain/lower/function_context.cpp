// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

#include "common/vlog.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {

FunctionContext::FunctionContext(FileContext& file_context,
                                 llvm::Function* function,
                                 llvm::raw_ostream* vlog_stream)
    : file_context_(&file_context),
      function_(function),
      builder_(file_context.llvm_context()),
      vlog_stream_(vlog_stream) {}

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

auto FunctionContext::LowerBlock(SemIR::NodeBlockId block_id) -> void {
  for (const auto& node_id : semantics_ir().GetNodeBlock(block_id)) {
    auto node = semantics_ir().GetNode(node_id);
    CARBON_VLOG() << "Lowering " << node_id << ": " << node << "\n";
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (node.kind()) {
#define CARBON_SEM_IR_NODE_KIND(Name)                     \
  case SemIR::Name::Kind:                                 \
    Handle##Name(*this, node_id, node.As<SemIR::Name>()); \
    break;
#include "toolchain/sem_ir/node_kind.def"
    }
  }
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
    case SemIR::ValueRepresentation::Unknown:
      CARBON_FATAL() << "Attempt to copy incomplete type";
    case SemIR::ValueRepresentation::None:
      break;
    case SemIR::ValueRepresentation::Copy:
      builder().CreateStore(GetLocal(source_id), GetLocal(dest_id));
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

}  // namespace Carbon::Lower
