// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <optional>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/formatter.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

Context::Context(DiagnosticEmitter* emitter,
                 Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter,
                 SemIR::File* sem_ir, int imported_ir_count, int total_ir_count,
                 llvm::raw_ostream* vlog_stream)
    : emitter_(emitter),
      tree_and_subtrees_getter_(tree_and_subtrees_getter),
      sem_ir_(sem_ir),
      vlog_stream_(vlog_stream),
      node_stack_(sem_ir->parse_tree(), vlog_stream),
      inst_block_stack_("inst_block_stack_", *sem_ir, vlog_stream),
      pattern_block_stack_("pattern_block_stack_", *sem_ir, vlog_stream),
      param_and_arg_refs_stack_(*sem_ir, vlog_stream, node_stack_),
      args_type_info_stack_("args_type_info_stack_", *sem_ir, vlog_stream),
      decl_name_stack_(this),
      scope_stack_(sem_ir_),
      vtable_stack_("vtable_stack_", *sem_ir, vlog_stream),
      global_init_(this),
      region_stack_(
          [this](SemIRLoc loc, std::string label) { TODO(loc, label); }) {
  // Prepare fields which relate to the number of IRs available for import.
  import_irs().Reserve(imported_ir_count);
  import_ir_constant_values_.reserve(imported_ir_count);
  check_ir_map_.resize(total_ir_count, SemIR::ImportIRId::None);

  // Map the builtin `<error>` and `type` type constants to their corresponding
  // special `TypeId` values.
  type_ids_for_type_constants_.Insert(
      SemIR::ConstantId::ForConcreteConstant(SemIR::ErrorInst::SingletonInstId),
      SemIR::ErrorInst::SingletonTypeId);
  type_ids_for_type_constants_.Insert(
      SemIR::ConstantId::ForConcreteConstant(SemIR::TypeType::SingletonInstId),
      SemIR::TypeType::SingletonTypeId);

  // TODO: Remove this and add a `VerifyOnFinish` once we properly push and pop
  // in the right places.
  generic_region_stack().Push();
}

auto Context::TODO(SemIRLoc loc, std::string label) -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "semantics TODO: `{0}`", std::string);
  emitter_->Emit(loc, SemanticsTodo, std::move(label));
  return false;
}

auto Context::VerifyOnFinish() -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  inst_block_stack_.VerifyOnFinish();
  pattern_block_stack_.VerifyOnFinish();
  param_and_arg_refs_stack_.VerifyOnFinish();
  args_type_info_stack_.VerifyOnFinish();
  CARBON_CHECK(struct_type_fields_stack_.empty());
  // TODO: Add verification for decl_name_stack_ and
  // decl_introducer_state_stack_.
  scope_stack_.VerifyOnFinish();
  // TODO: Add verification for generic_region_stack_.
}

auto Context::GetOrAddInst(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  if (loc_id_and_inst.loc_id.is_implicit()) {
    auto const_id =
        TryEvalInst(*this, SemIR::InstId::None, loc_id_and_inst.inst);
    if (const_id.has_value()) {
      CARBON_VLOG("GetOrAddInst: constant: {0}\n", loc_id_and_inst.inst);
      return constant_values().GetInstId(const_id);
    }
  }
  // TODO: For an implicit instruction, this reattempts evaluation.
  return AddInst(loc_id_and_inst);
}

// Finish producing an instruction. Set its constant value, and register it in
// any applicable instruction lists.
auto Context::FinishInst(SemIR::InstId inst_id, SemIR::Inst inst) -> void {
  GenericRegionStack::DependencyKind dep_kind =
      GenericRegionStack::DependencyKind::None;

  // If the instruction has a symbolic constant type, track that we need to
  // substitute into it.
  if (constant_values().DependsOnGenericParameter(
          types().GetConstantId(inst.type_id()))) {
    dep_kind |= GenericRegionStack::DependencyKind::SymbolicType;
  }

  // If the instruction has a constant value, compute it.
  auto const_id = TryEvalInst(*this, inst_id, inst);
  constant_values().Set(inst_id, const_id);
  if (const_id.is_constant()) {
    CARBON_VLOG("Constant: {0} -> {1}\n", inst,
                constant_values().GetInstId(const_id));

    // If the constant value is symbolic, track that we need to substitute into
    // it.
    if (constant_values().DependsOnGenericParameter(const_id)) {
      dep_kind |= GenericRegionStack::DependencyKind::SymbolicConstant;
    }
  }

  // Keep track of dependent instructions.
  if (dep_kind != GenericRegionStack::DependencyKind::None) {
    // TODO: Also check for template-dependent instructions.
    generic_region_stack().AddDependentInst(
        {.inst_id = inst_id, .kind = dep_kind});
  }
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

auto Context::CheckCompatibleImportedNodeKind(
    SemIR::ImportIRInstId imported_loc_id, SemIR::InstKind kind) -> void {
  auto& import_ir_inst = import_ir_insts().Get(imported_loc_id);
  const auto* import_ir = import_irs().Get(import_ir_inst.ir_id).sem_ir;
  auto imported_kind = import_ir->insts().Get(import_ir_inst.inst_id).kind();
  CARBON_CHECK(
      HasCompatibleImportedNodeKind(imported_kind, kind),
      "Node of kind {0} created with location of imported node of kind {1}",
      kind, imported_kind);
}

auto Context::AddPlaceholderInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG("AddPlaceholderInst: {0}\n", loc_id_and_inst.inst);
  constant_values().Set(inst_id, SemIR::ConstantId::None);
  return inst_id;
}

auto Context::AddPlaceholderInst(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddPlaceholderInstInNoBlock(loc_id_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::ReplaceLocIdAndInstBeforeConstantUse(
    SemIR::InstId inst_id, SemIR::LocIdAndInst loc_id_and_inst) -> void {
  sem_ir().insts().SetLocIdAndInst(inst_id, loc_id_and_inst);
  CARBON_VLOG("ReplaceInst: {0} -> {1}\n", inst_id, loc_id_and_inst.inst);
  FinishInst(inst_id, loc_id_and_inst.inst);
}

auto Context::ReplaceInstBeforeConstantUse(SemIR::InstId inst_id,
                                           SemIR::Inst inst) -> void {
  sem_ir().insts().Set(inst_id, inst);
  CARBON_VLOG("ReplaceInst: {0} -> {1}\n", inst_id, inst);
  FinishInst(inst_id, inst);
}

auto Context::ReplaceInstPreservingConstantValue(SemIR::InstId inst_id,
                                                 SemIR::Inst inst) -> void {
  auto old_const_id = sem_ir().constant_values().Get(inst_id);
  sem_ir().insts().Set(inst_id, inst);
  CARBON_VLOG("ReplaceInst: {0} -> {1}\n", inst_id, inst);
  auto new_const_id = TryEvalInst(*this, inst_id, inst);
  CARBON_CHECK(old_const_id == new_const_id);
}

auto Context::Finalize() -> void {
  // Pop information for the file-level scope.
  sem_ir().set_top_inst_block_id(inst_block_stack().Pop());
  scope_stack().Pop();

  // Finalizes the list of exports on the IR.
  inst_blocks().Set(SemIR::InstBlockId::Exports, exports_);
  // Finalizes the ImportRef inst block.
  inst_blocks().Set(SemIR::InstBlockId::ImportRefs, import_ref_ids_);
  // Finalizes __global_init.
  global_init_.Finalize();
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  output << "Check::Context\n";

  // In a stack dump, this is probably indented by a tab. We treat that as 8
  // spaces then add a couple to indent past the Context label.
  constexpr int Indent = 10;

  node_stack_.PrintForStackDump(Indent, output);
  inst_block_stack_.PrintForStackDump(Indent, output);
  pattern_block_stack_.PrintForStackDump(Indent, output);
  param_and_arg_refs_stack_.PrintForStackDump(Indent, output);
  args_type_info_stack_.PrintForStackDump(Indent, output);
}

auto Context::DumpFormattedFile() const -> void {
  SemIR::Formatter formatter(sem_ir_);
  formatter.Print(llvm::errs());
}

}  // namespace Carbon::Check
