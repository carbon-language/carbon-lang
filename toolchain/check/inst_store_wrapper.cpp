// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/inst_store_wrapper.h"

#include "common/vlog.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"

namespace Carbon::Check {

auto InstStoreWrapper::insts() const -> const SemIR::InstStore& {
  return context_->sem_ir().insts();
}

auto InstStoreWrapper::Add(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddInNoBlock(loc_id_and_inst);
  context_->inst_block_stack().AddInstId(inst_id);
  return inst_id;
}

auto InstStoreWrapper::AddInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = context_->sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG_TO(context_->vlog_stream(), "insts.Add: {0}\n",
                 loc_id_and_inst.inst);
  Finish(inst_id, loc_id_and_inst.inst);
  return inst_id;
}

auto InstStoreWrapper::AddInPatternBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddInNoBlock(loc_id_and_inst);
  context_->pattern_block_stack().AddInstId(inst_id);
  return inst_id;
}

auto InstStoreWrapper::GetOrAdd(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  if (loc_id_and_inst.loc_id.is_implicit()) {
    auto const_id =
        TryEvalInst(*context_, SemIR::InstId::None, loc_id_and_inst.inst);
    if (const_id.has_value()) {
      CARBON_VLOG_TO(context_->vlog_stream(), "insts.GetOrAdd: constant: {0}\n",
                     loc_id_and_inst.inst);
      return context_->constant_values().GetInstId(const_id);
    }
  }
  // TODO: For an implicit instruction, this reattempts evaluation.
  return Add(loc_id_and_inst);
}

auto InstStoreWrapper::AddPlaceholderInNoBlock(
    SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId {
  auto inst_id = context_->sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG_TO(context_->vlog_stream(), "insts.AddPlaceholder: {0}\n",
                 loc_id_and_inst.inst);
  context_->constant_values().Set(inst_id, SemIR::ConstantId::None);
  return inst_id;
}

auto InstStoreWrapper::AddPlaceholder(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddPlaceholderInNoBlock(loc_id_and_inst);
  context_->inst_block_stack().AddInstId(inst_id);
  return inst_id;
}

auto InstStoreWrapper::ReplaceLocIdAndInstBeforeConstantUse(
    SemIR::InstId inst_id, SemIR::LocIdAndInst loc_id_and_inst) -> void {
  context_->sem_ir().insts().SetLocIdAndInst(inst_id, loc_id_and_inst);
  CARBON_VLOG_TO(context_->vlog_stream(), "insts.Replace: {0} -> {1}\n",
                 inst_id, loc_id_and_inst.inst);
  Finish(inst_id, loc_id_and_inst.inst);
}

auto InstStoreWrapper::ReplaceBeforeConstantUse(SemIR::InstId inst_id,
                                                SemIR::Inst inst) -> void {
  context_->sem_ir().insts().Set(inst_id, inst);
  CARBON_VLOG_TO(context_->vlog_stream(), "insts.Replace: {0} -> {1}\n",
                 inst_id, inst);
  Finish(inst_id, inst);
}

auto InstStoreWrapper::ReplacePreservingConstantValue(SemIR::InstId inst_id,
                                                      SemIR::Inst inst)
    -> void {
  auto old_const_id = context_->constant_values().Get(inst_id);
  context_->sem_ir().insts().Set(inst_id, inst);
  CARBON_VLOG_TO(context_->vlog_stream(), "insts.Replace: {0} -> {1}\n",
                 inst_id, inst);
  auto new_const_id = TryEvalInst(*context_, inst_id, inst);
  CARBON_CHECK(old_const_id == new_const_id);
}

auto InstStoreWrapper::SetNamespaceNodeId(SemIR::InstId inst_id,
                                          Parse::NodeId node_id) -> void {
  context_->sem_ir().insts().SetLocId(inst_id, SemIR::LocId(node_id));
}

// Returns whether a parse node associated with an imported instruction of kind
// `imported_kind` is usable as the location of a corresponding local
// instruction of kind `local_kind`.
static auto HasCompatibleImportedNodeKind(SemIR::InstKind imported_kind,
                                          SemIR::InstKind local_kind) -> bool {
  if (imported_kind == local_kind) {
    return true;
  }
  if (imported_kind == SemIR::ImportDecl::Kind &&
      local_kind == SemIR::Namespace::Kind) {
    static_assert(
        std::is_convertible_v<decltype(SemIR::ImportDecl::Kind)::TypedNodeId,
                              decltype(SemIR::Namespace::Kind)::TypedNodeId>);
    return true;
  }
  return false;
}

auto InstStoreWrapper::CheckCompatibleImportedNodeKind(
    SemIR::ImportIRInstId imported_loc_id, SemIR::InstKind kind) -> void {
  auto& import_ir_inst = context_->import_ir_insts().Get(imported_loc_id);
  const auto* import_ir =
      context_->import_irs().Get(import_ir_inst.ir_id).sem_ir;
  auto imported_kind = import_ir->insts().Get(import_ir_inst.inst_id).kind();
  CARBON_CHECK(
      HasCompatibleImportedNodeKind(imported_kind, kind),
      "Node of kind {0} created with location of imported node of kind {1}",
      kind, imported_kind);
}

auto InstStoreWrapper::Finish(SemIR::InstId inst_id, SemIR::Inst inst) -> void {
  GenericRegionStack::DependencyKind dep_kind =
      GenericRegionStack::DependencyKind::None;

  // If the instruction has a symbolic constant type, track that we need to
  // substitute into it.
  if (context_->constant_values().DependsOnGenericParameter(
          context_->types().GetConstantId(inst.type_id()))) {
    dep_kind |= GenericRegionStack::DependencyKind::SymbolicType;
  }

  // If the instruction has a constant value, compute it.
  auto const_id = TryEvalInst(*context_, inst_id, inst);
  context_->constant_values().Set(inst_id, const_id);
  if (const_id.is_constant()) {
    CARBON_VLOG_TO(context_->vlog_stream(), "Constant: {0} -> {1}\n", inst,
                   context_->constant_values().GetInstId(const_id));

    // If the constant value is symbolic, track that we need to substitute into
    // it.
    if (context_->constant_values().DependsOnGenericParameter(const_id)) {
      dep_kind |= GenericRegionStack::DependencyKind::SymbolicConstant;
    }
  }

  // Keep track of dependent instructions.
  if (dep_kind != GenericRegionStack::DependencyKind::None) {
    // TODO: Also check for template-dependent instructions.
    context_->generic_region_stack().AddDependentInst(
        {.inst_id = inst_id, .kind = dep_kind});
  }
}

auto InstStoreWrapper::NodeStackPush(Parse::NodeId node_id,
                                     SemIR::InstId inst_id) -> void {
  context_->node_stack().Push(node_id, inst_id);
}

}  // namespace Carbon::Check
