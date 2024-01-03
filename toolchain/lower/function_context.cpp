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

auto FunctionContext::GetBlock(SemIR::InstBlockId block_id)
    -> llvm::BasicBlock* {
  llvm::BasicBlock*& entry = blocks_[block_id];
  if (!entry) {
    entry = llvm::BasicBlock::Create(llvm_context(), "", function_);
  }
  return entry;
}

auto FunctionContext::TryToReuseBlock(SemIR::InstBlockId block_id,
                                      llvm::BasicBlock* block) -> bool {
  if (!blocks_.insert({block_id, block}).second) {
    return false;
  }
  if (block == synthetic_block_) {
    synthetic_block_ = nullptr;
  }
  return true;
}

// Structure for the core handler dispatch.
using DispatchFunctionT = auto(FunctionContext& context, SemIR::InstId inst_id,
                               SemIR::Inst inst) -> void;
using DispatchTableT =
    std::array<DispatchFunctionT*, SemIR::InstKind::EnumCount>;

// Transforms a parse node ID to a typed parse node, which is then passed to the
// handler.
#define CARBON_SEM_IR_INST_KIND(Name)                                  \
  auto Dispatch##Name(FunctionContext& context, SemIR::InstId inst_id, \
                      SemIR::Inst inst) -> void {                      \
    return Handle##Name(context, inst_id, inst.As<SemIR::Name>());     \
  }
#include "toolchain/sem_ir/inst_kind.def"

// The main dispatch table. This is used instead of a switch in order to
// optimize for common functionality.
static constexpr DispatchTableT DispatchTable = {
#define CARBON_SEM_IR_INST_KIND(Name) &Dispatch##Name,
#include "toolchain/sem_ir/inst_kind.def"
};

auto FunctionContext::LowerBlock(SemIR::InstBlockId block_id) -> void {
  for (const auto& inst_id : sem_ir().inst_blocks().Get(block_id)) {
    auto inst = sem_ir().insts().Get(inst_id);
    CARBON_VLOG() << "Lowering " << inst_id << ": " << inst << "\n";
    DispatchTable[inst.kind().AsInt()](*this, inst_id, inst);
  }
}

auto FunctionContext::GetBlockArg(SemIR::InstBlockId block_id,
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

auto FunctionContext::FinishInit(SemIR::TypeId type_id, SemIR::InstId dest_id,
                                 SemIR::InstId source_id) -> void {
  switch (SemIR::GetInitRepr(sem_ir(), type_id).kind) {
    case SemIR::InitRepr::None:
    case SemIR::InitRepr::InPlace:
      break;
    case SemIR::InitRepr::ByCopy:
      CopyValue(type_id, source_id, dest_id);
      break;
  }
}

auto FunctionContext::CopyValue(SemIR::TypeId type_id, SemIR::InstId source_id,
                                SemIR::InstId dest_id) -> void {
  switch (auto rep = SemIR::GetValueRepr(sem_ir(), type_id); rep.kind) {
    case SemIR::ValueRepr::Unknown:
      CARBON_FATAL() << "Attempt to copy incomplete type";
    case SemIR::ValueRepr::None:
      break;
    case SemIR::ValueRepr::Copy:
      builder().CreateStore(GetValue(source_id), GetValue(dest_id));
      break;
    case SemIR::ValueRepr::Pointer: {
      const auto& layout = llvm_module().getDataLayout();
      auto* type = GetType(type_id);
      // TODO: Compute known alignment of the source and destination, which may
      // be greater than the alignment computed by LLVM.
      auto align = layout.getABITypeAlign(type);

      // TODO: Attach !tbaa.struct metadata indicating which portions of the
      // type we actually need to copy and which are padding.
      builder().CreateMemCpy(GetValue(dest_id), align, GetValue(source_id),
                             align, layout.getTypeAllocSize(type));
      break;
    }
    case SemIR::ValueRepr::Custom:
      CARBON_FATAL() << "TODO: Add support for CopyValue with custom value rep";
  }
}

}  // namespace Carbon::Lower
