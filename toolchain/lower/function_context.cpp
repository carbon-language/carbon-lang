// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

#include "common/vlog.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"

namespace Carbon::Lower {

FunctionContext::FunctionContext(
    FileContext& file_context, llvm::Function* function,
    SemIR::SpecificId specific_id,
    FileContext::SpecificFunctionFingerprint* function_fingerprint,
    llvm::DISubprogram* di_subprogram, llvm::raw_ostream* vlog_stream)
    : file_context_(&file_context),
      function_(function),
      specific_id_(specific_id),
      builder_(file_context.llvm_context(), llvm::ConstantFolder(),
               Inserter(file_context.inst_namer())),
      di_subprogram_(di_subprogram),
      vlog_stream_(vlog_stream),
      function_fingerprint_(function_fingerprint) {
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
                            InstT inst) -> void {
  if constexpr (!InstT::Kind.is_lowered()) {
    CARBON_FATAL(
        "Encountered an instruction that isn't expected to lower. It's "
        "possible that logic needs to be changed in order to stop showing this "
        "instruction in lowered contexts. Instruction: {0}",
        inst);
  } else if constexpr (InstT::Kind.constant_kind() ==
                           SemIR::InstConstantKind::Always ||
                       InstT::Kind.constant_kind() ==
                           SemIR::InstConstantKind::AlwaysUnique) {
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

  auto debug_loc = GetDebugLoc(inst_id);
  if (debug_loc) {
    builder_.SetCurrentDebugLocation(debug_loc);
  }

  CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(Name)            \
  case CARBON_KIND(SemIR::Name typed_inst): {    \
    LowerInstHelper(*this, inst_id, typed_inst); \
    break;                                       \
  }
#include "toolchain/sem_ir/inst_kind.def"
  }

  if (debug_loc) {
    builder_.SetCurrentDebugLocation(llvm::DebugLoc());
  }

  builder_.getInserter().SetCurrentInstId(SemIR::InstId::None);
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

auto FunctionContext::GetValue(SemIR::InstId inst_id) -> llvm::Value* {
  // All builtins are types, with the same empty lowered value.
  if (SemIR::IsSingletonInstId(inst_id)) {
    return GetTypeAsValue();
  }

  if (auto result = locals_.Lookup(inst_id)) {
    return result.value();
  }

  if (auto result = file_context_->global_variables().Lookup(inst_id)) {
    return result.value();
  }

  auto* global = file_context_->GetGlobal(inst_id, specific_id_);
  AddGlobalToCurrentFingerprint(global);
  return global;
}

auto FunctionContext::MakeSyntheticBlock() -> llvm::BasicBlock* {
  synthetic_block_ = llvm::BasicBlock::Create(llvm_context(), "", function_);
  return synthetic_block_;
}

auto FunctionContext::GetDebugLoc(SemIR::InstId inst_id) -> llvm::DebugLoc {
  if (!di_subprogram_) {
    return llvm::DebugLoc();
  }
  auto loc = file_context_->GetLocForDI(inst_id);
  if (loc.filename != di_subprogram_->getFile()->getFilename()) {
    // Location is from a different file. We can't represent that directly
    // within the scope of this function's subprogram, and we don't want to
    // generate a new subprogram, so just discard the location information. This
    // happens for thunks when emitting the portion of the thunk that is
    // duplicated from the original signature.
    //
    // TODO: Handle this case better.
    return llvm::DebugLoc();
  }
  return llvm::DILocation::get(builder_.getContext(), loc.line_number,
                               loc.column_number, di_subprogram_);
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

auto FunctionContext::GetTypeOfInst(SemIR::InstId inst_id) -> SemIR::TypeId {
  return SemIR::GetTypeOfInstInSpecific(sem_ir(), specific_id(), inst_id);
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

auto FunctionContext::AddCallToCurrentFingerprint(SemIR::FunctionId function_id,
                                                  SemIR::SpecificId specific_id)
    -> void {
  if (!function_fingerprint_) {
    return;
  }

  RawStringOstream os;
  // TODO: Replace index with info that is translation unit independent.
  // Using a string that includes the `FunctionId` string and the index to
  // avoid possible collisions. This needs revisiting.
  os << "function_id" << function_id.index << "\n";
  current_fingerprint_.common_fingerprint.update(os.TakeStr());
  // TODO: Replace index with info that is translation unit independent.
  if (specific_id.has_value()) {
    current_fingerprint_.specific_fingerprint.update(specific_id.index);
    // TODO: Uses -1 as delimiter. This needs revisiting.
    current_fingerprint_.specific_fingerprint.update(-1);
    function_fingerprint_->calls.push_back(specific_id);
  }
}

auto FunctionContext::AddTypeToCurrentFingerprint(llvm::Type* type) -> void {
  if (!function_fingerprint_ || !type) {
    return;
  }

  RawStringOstream os;
  type->print(os);
  os << "\n";
  current_fingerprint_.common_fingerprint.update(os.TakeStr());
}

auto FunctionContext::AddGlobalToCurrentFingerprint(llvm::Value* global)
    -> void {
  if (!function_fingerprint_ || !global) {
    return;
  }

  RawStringOstream os;
  global->print(os);
  os << "\n";
  current_fingerprint_.common_fingerprint.update(os.TakeStr());
}

auto FunctionContext::EmitFinalFingerprint() -> void {
  if (!function_fingerprint_) {
    return;
  }
  current_fingerprint_.common_fingerprint.final(
      function_fingerprint_->common_fingerprint);
  current_fingerprint_.specific_fingerprint.final(
      function_fingerprint_->specific_fingerprint);
}

}  // namespace Carbon::Lower
