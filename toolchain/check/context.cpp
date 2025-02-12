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

auto Context::DiagnoseDuplicateName(SemIRLoc dup_def, SemIRLoc prev_def)
    -> void {
  CARBON_DIAGNOSTIC(NameDeclDuplicate, Error,
                    "duplicate name being declared in the same scope");
  CARBON_DIAGNOSTIC(NameDeclPrevious, Note, "name is previously declared here");
  emitter_->Build(dup_def, NameDeclDuplicate)
      .Note(prev_def, NameDeclPrevious)
      .Emit();
}

auto Context::DiagnosePoisonedName(SemIR::LocId poisoning_loc_id,
                                   SemIR::LocId decl_name_loc_id) -> void {
  CARBON_CHECK(poisoning_loc_id.has_value(),
               "Trying to diagnose poisoned name with no poisoning location");
  CARBON_DIAGNOSTIC(NameUseBeforeDecl, Error,
                    "name used before it was declared");
  CARBON_DIAGNOSTIC(NameUseBeforeDeclNote, Note, "declared here");
  emitter_->Build(poisoning_loc_id, NameUseBeforeDecl)
      .Note(decl_name_loc_id, NameUseBeforeDeclNote)
      .Emit();
}

auto Context::DiagnoseNameNotFound(SemIRLoc loc, SemIR::NameId name_id)
    -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "name `{0}` not found", SemIR::NameId);
  emitter_->Emit(loc, NameNotFound, name_id);
}

auto Context::NoteAbstractClass(SemIR::ClassId class_id,
                                DiagnosticBuilder& builder) -> void {
  const auto& class_info = classes().Get(class_id);
  CARBON_CHECK(
      class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract,
      "Class is not abstract");
  CARBON_DIAGNOSTIC(ClassAbstractHere, Note,
                    "class was declared abstract here");
  builder.Note(class_info.definition_id, ClassAbstractHere);
}

auto Context::NoteIncompleteClass(SemIR::ClassId class_id,
                                  DiagnosticBuilder& builder) -> void {
  const auto& class_info = classes().Get(class_id);
  CARBON_CHECK(!class_info.is_defined(), "Class is not incomplete");
  if (class_info.has_definition_started()) {
    CARBON_DIAGNOSTIC(ClassIncompleteWithinDefinition, Note,
                      "class is incomplete within its definition");
    builder.Note(class_info.definition_id, ClassIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                      "class was forward declared here");
    builder.Note(class_info.latest_decl_id(), ClassForwardDeclaredHere);
  }
}

auto Context::NoteUndefinedInterface(SemIR::InterfaceId interface_id,
                                     DiagnosticBuilder& builder) -> void {
  const auto& interface_info = interfaces().Get(interface_id);
  CARBON_CHECK(!interface_info.is_defined(), "Interface is not incomplete");
  if (interface_info.is_being_defined()) {
    CARBON_DIAGNOSTIC(InterfaceUndefinedWithinDefinition, Note,
                      "interface is currently being defined");
    builder.Note(interface_info.definition_id,
                 InterfaceUndefinedWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(InterfaceForwardDeclaredHere, Note,
                      "interface was forward declared here");
    builder.Note(interface_info.latest_decl_id(), InterfaceForwardDeclaredHere);
  }
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

auto Context::GetTypeIdForTypeConstant(SemIR::ConstantId constant_id)
    -> SemIR::TypeId {
  CARBON_CHECK(constant_id.is_constant(),
               "Canonicalizing non-constant type: {0}", constant_id);
  auto type_id =
      insts().Get(constant_values().GetInstId(constant_id)).type_id();
  CARBON_CHECK(type_id == SemIR::TypeType::SingletonTypeId ||
                   constant_id == SemIR::ErrorInst::SingletonConstantId,
               "Forming type ID for non-type constant of type {0}",
               types().GetAsInst(type_id));

  return SemIR::TypeId::ForTypeConstant(constant_id);
}

auto Context::FacetTypeFromInterface(SemIR::InterfaceId interface_id,
                                     SemIR::SpecificId specific_id)
    -> SemIR::FacetType {
  SemIR::FacetTypeId facet_type_id = facet_types().Add(
      SemIR::FacetTypeInfo{.impls_constraints = {{interface_id, specific_id}},
                           .other_requirements = false});
  return {.type_id = SemIR::TypeType::SingletonTypeId,
          .facet_type_id = facet_type_id};
}

// Gets or forms a type_id for a type, given the instruction kind and arguments.
template <typename InstT, typename... EachArgT>
static auto GetTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  // TODO: Remove inst_id parameter from TryEvalInst.
  InstT inst = {SemIR::TypeType::SingletonTypeId, each_arg...};
  return context.GetTypeIdForTypeConstant(
      TryEvalInst(context, SemIR::InstId::None, inst));
}

// Gets or forms a type_id for a type, given the instruction kind and arguments,
// and completes the type. This should only be used when type completion cannot
// fail.
template <typename InstT, typename... EachArgT>
static auto GetCompleteTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  auto type_id = GetTypeImpl<InstT>(context, each_arg...);
  CompleteTypeOrCheckFail(context, type_id);
  return type_id;
}

auto Context::GetStructType(SemIR::StructTypeFieldsId fields_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::StructType>(*this, fields_id);
}

auto Context::GetTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::TupleType>(*this,
                                       type_blocks().AddCanonical(type_ids));
}

auto Context::GetAssociatedEntityType(SemIR::TypeId interface_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::AssociatedEntityType>(*this, interface_type_id);
}

auto Context::GetSingletonType(SemIR::InstId singleton_id) -> SemIR::TypeId {
  CARBON_CHECK(SemIR::IsSingletonInstId(singleton_id));
  auto type_id = GetTypeIdForTypeInst(singleton_id);
  // To keep client code simpler, complete builtin types before returning them.
  CompleteTypeOrCheckFail(*this, type_id);
  return type_id;
}

auto Context::GetClassType(SemIR::ClassId class_id,
                           SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::ClassType>(*this, class_id, specific_id);
}

auto Context::GetFunctionType(SemIR::FunctionId fn_id,
                              SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::FunctionType>(*this, fn_id, specific_id);
}

auto Context::GetFunctionTypeWithSelfType(
    SemIR::InstId interface_function_type_id, SemIR::InstId self_id)
    -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::FunctionTypeWithSelfType>(
      *this, interface_function_type_id, self_id);
}

auto Context::GetGenericClassType(SemIR::ClassId class_id,
                                  SemIR::SpecificId enclosing_specific_id)
    -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::GenericClassType>(*this, class_id,
                                                      enclosing_specific_id);
}

auto Context::GetGenericInterfaceType(SemIR::InterfaceId interface_id,
                                      SemIR::SpecificId enclosing_specific_id)
    -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::GenericInterfaceType>(
      *this, interface_id, enclosing_specific_id);
}

auto Context::GetInterfaceType(SemIR::InterfaceId interface_id,
                               SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::FacetType>(
      *this, FacetTypeFromInterface(interface_id, specific_id).facet_type_id);
}

auto Context::GetPointerType(SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::PointerType>(*this, pointee_type_id);
}

auto Context::GetUnboundElementType(SemIR::TypeId class_type_id,
                                    SemIR::TypeId element_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::UnboundElementType>(*this, class_type_id,
                                                element_type_id);
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
