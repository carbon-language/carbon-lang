// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

#include "common/vlog.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {

FunctionContext::FunctionContext(FileContext& file_context,
                                 llvm::Function* function,
                                 llvm::DISubprogram* di_subprogram,
                                 llvm::raw_ostream* vlog_stream)
    : file_context_(&file_context),
      function_(function),
      builder_(file_context.llvm_context(), llvm::ConstantFolder(),
               Inserter(file_context.inst_namer())),
      di_subprogram_(di_subprogram),
      vlog_stream_(vlog_stream) {
  function_->setSubprogram(di_subprogram_);
}

auto FunctionContext::GetBlock(SemIR::InstBlockId block_id)
    -> llvm::BasicBlock* {
  auto result = blocks_.Insert(block_id, [&] {
    llvm::StringRef label_name;
    if (const auto* inst_namer = file_context_->inst_namer()) {
      label_name = inst_namer->GetUnscopedLabelFor(block_id);
    }
    return llvm::BasicBlock::Create(llvm_context(), label_name, function_);
  });
  return result.value();
}

auto FunctionContext::TryToReuseBlock(SemIR::InstBlockId block_id,
                                      llvm::BasicBlock* block) -> bool {
  if (!blocks_.Insert(block_id, block).is_inserted()) {
    return false;
  }
  if (block == synthetic_block_) {
    synthetic_block_ = nullptr;
  }
  if (const auto* inst_namer = file_context_->inst_namer()) {
    block->setName(inst_namer->GetUnscopedLabelFor(block_id));
  }
  return true;
}

auto FunctionContext::LowerBlockContents(SemIR::InstBlockId block_id) -> void {
  for (auto inst_id : sem_ir().inst_blocks().Get(block_id)) {
    LowerInst(inst_id);
  }
}

// Handles typed instructions for LowerInst. Many instructions lower using
// HandleInst, but others are unsupported or have trivial lowering.
//
// This only calls HandleInst for versions that should have implementations. A
// different approach would be to have the logic below implemented as HandleInst
// overloads. However, forward declarations of HandleInst exist for all `InstT`
// types, which would make getting the right overload resolution complex.
template <typename InstT>
static auto LowerInstHelper(FunctionContext& context, SemIR::InstId inst_id,
                            InstT inst) {
  if constexpr (!InstT::Kind.is_lowered()) {
    CARBON_FATAL(
        "Encountered an instruction that isn't expected to lower. It's "
        "possible that logic needs to be changed in order to stop showing this "
        "instruction in lowered contexts. Instruction: {0}",
        inst);
  } else if constexpr (InstT::Kind.constant_kind() ==
                       SemIR::InstConstantKind::Always) {
    CARBON_FATAL("Missing constant value for constant instruction {0}", inst);
  } else if constexpr (InstT::Kind.is_type() == SemIR::InstIsType::Always) {
    // For instructions that are always of type `type`, produce the trivial
    // runtime representation of type `type`.
    context.SetLocal(inst_id, context.GetTypeAsValue());
  } else {
    HandleInst(context, inst_id, inst);
  }
}

// TODO: Consider renaming Handle##Name, instead relying on typed_inst overload
// resolution. That would allow putting the nonexistent handler implementations
// in `requires`-style overloads.
// NOLINTNEXTLINE(readability-function-size): The define confuses lint.
auto FunctionContext::LowerInst(SemIR::InstId inst_id) -> void {
  // Skip over constants. `FileContext::GetGlobal` lowers them as needed.
  if (sem_ir().constant_values().Get(inst_id).is_constant()) {
    return;
  }

  auto inst = sem_ir().insts().Get(inst_id);
  CARBON_VLOG("Lowering {0}: {1}\n", inst_id, inst);
  builder_.getInserter().SetCurrentInstId(inst_id);
  if (di_subprogram_) {
    auto loc = file_context_->GetLocForDI(inst_id);
    CARBON_CHECK(loc.filename == di_subprogram_->getFile()->getFilename(),
                 "Instructions located in a different file from their "
                 "enclosing function aren't handled yet");
    builder_.SetCurrentDebugLocation(
        llvm::DILocation::get(builder_.getContext(), loc.line_number,
                              loc.column_number, di_subprogram_));
  }

  CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(Name)            \
  case CARBON_KIND(SemIR::Name typed_inst): {    \
    LowerInstHelper(*this, inst_id, typed_inst); \
    break;                                       \
  }
#include "toolchain/sem_ir/inst_kind.def"
  }

  builder_.getInserter().SetCurrentInstId(SemIR::InstId::None);
  if (di_subprogram_) {
    builder_.SetCurrentDebugLocation(llvm::DebugLoc());
  }
}

auto FunctionContext::GetBlockArg(SemIR::InstBlockId block_id,
                                  SemIR::TypeId type_id) -> llvm::PHINode* {
  llvm::BasicBlock* block = GetBlock(block_id);

  // Find the existing phi, if any.
  auto phis = block->phis();
  if (!phis.empty()) {
    CARBON_CHECK(std::next(phis.begin()) == phis.end(),
                 "Expected at most one phi, found {0}",
                 std::distance(phis.begin(), phis.end()));
    return &*phis.begin();
  }

  // The number of predecessor slots to reserve.
  static constexpr unsigned NumReservedPredecessors = 2;
  auto* phi = llvm::PHINode::Create(GetType(type_id), NumReservedPredecessors);
  phi->insertInto(block, block->begin());
  return phi;
}

auto FunctionContext::MakeSyntheticBlock() -> llvm::BasicBlock* {
  synthetic_block_ = llvm::BasicBlock::Create(llvm_context(), "", function_);
  return synthetic_block_;
}

auto FunctionContext::FinishInit(SemIR::TypeId type_id, SemIR::InstId dest_id,
                                 SemIR::InstId source_id) -> void {
  switch (SemIR::InitRepr::ForType(sem_ir(), type_id).kind) {
    case SemIR::InitRepr::None:
      break;
    case SemIR::InitRepr::InPlace:
      if (sem_ir().constant_values().Get(source_id).is_constant()) {
        // When initializing from a constant, emission of the source doesn't
        // initialize the destination. Copy the constant value instead.
        CopyValue(type_id, source_id, dest_id);
      }
      break;
    case SemIR::InitRepr::ByCopy:
      CopyValue(type_id, source_id, dest_id);
      break;
    case SemIR::InitRepr::Incomplete:
      CARBON_FATAL("Lowering aggregate initialization of incomplete type {0}",
                   sem_ir().types().GetAsInst(type_id));
  }
}

auto FunctionContext::CopyValue(SemIR::TypeId type_id, SemIR::InstId source_id,
                                SemIR::InstId dest_id) -> void {
  switch (auto rep = SemIR::ValueRepr::ForType(sem_ir(), type_id); rep.kind) {
    case SemIR::ValueRepr::Unknown:
      CARBON_FATAL("Attempt to copy incomplete type");
    case SemIR::ValueRepr::None:
      break;
    case SemIR::ValueRepr::Copy:
      builder().CreateStore(GetValue(source_id), GetValue(dest_id));
      break;
    case SemIR::ValueRepr::Pointer:
      CopyObject(type_id, source_id, dest_id);
      break;
    case SemIR::ValueRepr::Custom:
      CARBON_FATAL("TODO: Add support for CopyValue with custom value rep");
  }
}

auto FunctionContext::CopyObject(SemIR::TypeId type_id, SemIR::InstId source_id,
                                 SemIR::InstId dest_id) -> void {
  const auto& layout = llvm_module().getDataLayout();
  auto* type = GetType(type_id);
  // TODO: Compute known alignment of the source and destination, which may
  // be greater than the alignment computed by LLVM.
  auto align = layout.getABITypeAlign(type);

  // TODO: Attach !tbaa.struct metadata indicating which portions of the
  // type we actually need to copy and which are padding.
  builder().CreateMemCpy(GetValue(dest_id), align, GetValue(source_id), align,
                         layout.getTypeAllocSize(type));
}

auto FunctionContext::Inserter::InsertHelper(
    llvm::Instruction* inst, const llvm::Twine& name,
    llvm::BasicBlock::iterator insert_pt) const -> void {
  llvm::StringRef base_name;
  llvm::StringRef separator;
  if (inst_namer_ && !inst->getType()->isVoidTy()) {
    base_name = inst_namer_->GetUnscopedNameFor(inst_id_);
  }
  if (!base_name.empty() && !name.isTriviallyEmpty()) {
    separator = ".";
  }

  IRBuilderDefaultInserter::InsertHelper(inst, base_name + separator + name,
                                         insert_pt);
}

}  // namespace Carbon::Lower
