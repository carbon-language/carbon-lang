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
      scope_stack_(sem_ir_->identifiers()),
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
      SemIR::ConstantId::ForTemplateConstant(SemIR::ErrorInst::SingletonInstId),
      SemIR::ErrorInst::SingletonTypeId);
  type_ids_for_type_constants_.Insert(
      SemIR::ConstantId::ForTemplateConstant(SemIR::TypeType::SingletonInstId),
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
                                   SemIR::InstId decl_inst_id) -> void {
  CARBON_CHECK(poisoning_loc_id.has_value(),
               "Trying to diagnose poisoned name with no poisoning location");
  CARBON_DIAGNOSTIC(NameUseBeforeDecl, Error,
                    "name used before it was declared");
  CARBON_DIAGNOSTIC(NameUseBeforeDeclNote, Note, "declared here");
  emitter_->Build(poisoning_loc_id, NameUseBeforeDecl)
      .Note(decl_inst_id, NameUseBeforeDeclNote)
      .Emit();
}

auto Context::DiagnoseNameNotFound(SemIRLoc loc, SemIR::NameId name_id)
    -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "name `{0}` not found", SemIR::NameId);
  emitter_->Emit(loc, NameNotFound, name_id);
}

auto Context::DiagnoseMemberNameNotFound(
    SemIRLoc loc, SemIR::NameId name_id,
    llvm::ArrayRef<LookupScope> lookup_scopes) -> void {
  if (lookup_scopes.size() == 1 &&
      lookup_scopes.front().name_scope_id.has_value()) {
    auto specific_id = lookup_scopes.front().specific_id;
    auto scope_inst_id =
        specific_id.has_value()
            ? GetInstForSpecific(*this, specific_id)
            : name_scopes().Get(lookup_scopes.front().name_scope_id).inst_id();
    CARBON_DIAGNOSTIC(MemberNameNotFoundInScope, Error,
                      "member name `{0}` not found in {1}", SemIR::NameId,
                      InstIdAsType);
    emitter_->Emit(loc, MemberNameNotFoundInScope, name_id, scope_inst_id);
    return;
  }

  CARBON_DIAGNOSTIC(MemberNameNotFound, Error, "member name `{0}` not found",
                    SemIR::NameId);
  emitter_->Emit(loc, MemberNameNotFound, name_id);
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

auto Context::AddNameToLookup(SemIR::NameId name_id, SemIR::InstId target_id,
                              ScopeIndex scope_index) -> void {
  if (auto existing =
          scope_stack().LookupOrAddName(name_id, target_id, scope_index);
      existing.has_value()) {
    DiagnoseDuplicateName(target_id, existing);
  }
}

auto Context::LookupNameInDecl(SemIR::LocId loc_id, SemIR::NameId name_id,
                               SemIR::NameScopeId scope_id,
                               ScopeIndex scope_index)
    -> SemIR::ScopeLookupResult {
  if (!scope_id.has_value()) {
    // Look for a name in the specified scope or a scope nested within it only.
    // There are two cases where the name would be in an outer scope:
    //
    //  - The name is the sole component of the declared name:
    //
    //    class A;
    //    fn F() {
    //      class A;
    //    }
    //
    //    In this case, the inner A is not the same class as the outer A, so
    //    lookup should not find the outer A.
    //
    //  - The name is a qualifier of some larger declared name:
    //
    //    class A { class B; }
    //    fn F() {
    //      class A.B {}
    //    }
    //
    //    In this case, we're not in the correct scope to define a member of
    //    class A, so we should reject, and we achieve this by not finding the
    //    name A from the outer scope.
    //
    // There is also one case where the name would be in an inner scope:
    //
    //  - The name is redeclared by a parameter of the same entity:
    //
    //    fn F() {
    //      class C(C:! type);
    //    }
    //
    //    In this case, the class C is not a redeclaration of its parameter, but
    //    we find the parameter in order to diagnose a redeclaration error.
    return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        scope_stack().LookupInLexicalScopesWithin(name_id, scope_index),
        SemIR::AccessKind::Public);
  } else {
    // We do not look into `extend`ed scopes here. A qualified name in a
    // declaration must specify the exact scope in which the name was originally
    // introduced:
    //
    //    base class A { fn F(); }
    //    class B { extend base: A; }
    //
    //    // Error, no `F` in `B`.
    //    fn B.F() {}
    return LookupNameInExactScope(loc_id, name_id, scope_id,
                                  name_scopes().Get(scope_id),
                                  /*is_being_declared=*/true);
  }
}

auto Context::LookupUnqualifiedName(Parse::NodeId node_id,
                                    SemIR::NameId name_id, bool required)
    -> LookupResult {
  // TODO: Check for shadowed lookup results.

  // Find the results from ancestor lexical scopes. These will be combined with
  // results from non-lexical scopes such as namespaces and classes.
  auto [lexical_result, non_lexical_scopes] =
      scope_stack().LookupInLexicalScopes(name_id);

  // Walk the non-lexical scopes and perform lookups into each of them.
  for (auto [index, lookup_scope_id, specific_id] :
       llvm::reverse(non_lexical_scopes)) {
    if (auto non_lexical_result =
            LookupQualifiedName(node_id, name_id,
                                LookupScope{.name_scope_id = lookup_scope_id,
                                            .specific_id = specific_id},
                                /*required=*/false);
        non_lexical_result.scope_result.is_found()) {
      return non_lexical_result;
    }
  }

  if (lexical_result == SemIR::InstId::InitTombstone) {
    CARBON_DIAGNOSTIC(UsedBeforeInitialization, Error,
                      "`{0}` used before initialization", SemIR::NameId);
    emitter_->Emit(node_id, UsedBeforeInitialization, name_id);
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeError()};
  }

  if (lexical_result.has_value()) {
    // A lexical scope never needs an associated specific. If there's a
    // lexically enclosing generic, then it also encloses the point of use of
    // the name.
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeFound(
                lexical_result, SemIR::AccessKind::Public)};
  }

  // We didn't find anything at all.
  if (required) {
    DiagnoseNameNotFound(node_id, name_id);
  }

  return {.specific_id = SemIR::SpecificId::None,
          .scope_result = SemIR::ScopeLookupResult::MakeError()};
}

auto Context::LookupNameInExactScope(SemIR::LocId loc_id, SemIR::NameId name_id,
                                     SemIR::NameScopeId scope_id,
                                     SemIR::NameScope& scope,
                                     bool is_being_declared)
    -> SemIR::ScopeLookupResult {
  if (auto entry_id = is_being_declared
                          ? scope.Lookup(name_id)
                          : scope.LookupOrPoison(loc_id, name_id)) {
    auto lookup_result = scope.GetEntry(*entry_id).result;
    if (!lookup_result.is_poisoned()) {
      LoadImportRef(*this, lookup_result.target_inst_id());
    }
    return lookup_result;
  }

  if (!scope.import_ir_scopes().empty()) {
    // TODO: Enforce other access modifiers for imports.
    return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        ImportNameFromOtherPackage(*this, loc_id, scope_id,
                                   scope.import_ir_scopes(), name_id),
        SemIR::AccessKind::Public);
  }
  return SemIR::ScopeLookupResult::MakeNotFound();
}

// Prints diagnostics on invalid qualified name access.
static auto DiagnoseInvalidQualifiedNameAccess(Context& context, SemIRLoc loc,
                                               SemIR::InstId scope_result_id,
                                               SemIR::NameId name_id,
                                               SemIR::AccessKind access_kind,
                                               bool is_parent_access,
                                               AccessInfo access_info) -> void {
  auto class_type = context.insts().TryGetAs<SemIR::ClassType>(
      context.constant_values().GetInstId(access_info.constant_id));
  if (!class_type) {
    return;
  }

  // TODO: Support scoped entities other than just classes.
  const auto& class_info = context.classes().Get(class_type->class_id);

  auto parent_type_id = class_info.self_type_id;

  if (access_kind == SemIR::AccessKind::Private && is_parent_access) {
    if (auto base_type_id =
            class_info.GetBaseType(context.sem_ir(), class_type->specific_id);
        base_type_id.has_value()) {
      parent_type_id = base_type_id;
    } else if (auto adapted_type_id = class_info.GetAdaptedType(
                   context.sem_ir(), class_type->specific_id);
               adapted_type_id.has_value()) {
      parent_type_id = adapted_type_id;
    } else {
      CARBON_FATAL("Expected parent for parent access");
    }
  }

  CARBON_DIAGNOSTIC(
      ClassInvalidMemberAccess, Error,
      "cannot access {0:private|protected} member `{1}` of type {2}",
      BoolAsSelect, SemIR::NameId, SemIR::TypeId);
  CARBON_DIAGNOSTIC(ClassMemberDeclaration, Note, "declared here");
  context.emitter()
      .Build(loc, ClassInvalidMemberAccess,
             access_kind == SemIR::AccessKind::Private, name_id, parent_type_id)
      .Note(scope_result_id, ClassMemberDeclaration)
      .Emit();
}

// Returns whether the access is prohibited by the access modifiers.
static auto IsAccessProhibited(std::optional<AccessInfo> access_info,
                               SemIR::AccessKind access_kind,
                               bool is_parent_access) -> bool {
  if (!access_info) {
    return false;
  }

  switch (access_kind) {
    case SemIR::AccessKind::Public:
      return false;
    case SemIR::AccessKind::Protected:
      return access_info->highest_allowed_access == SemIR::AccessKind::Public;
    case SemIR::AccessKind::Private:
      return access_info->highest_allowed_access !=
                 SemIR::AccessKind::Private ||
             is_parent_access;
  }
}

// Information regarding a prohibited access.
struct ProhibitedAccessInfo {
  // The resulting inst of the lookup.
  SemIR::InstId scope_result_id;
  // The access kind of the lookup.
  SemIR::AccessKind access_kind;
  // If the lookup is from an extended scope. For example, if this is a base
  // class member access from a class that extends it.
  bool is_parent_access;
};

auto Context::AppendLookupScopesForConstant(
    SemIR::LocId loc_id, SemIR::ConstantId base_const_id,
    llvm::SmallVector<LookupScope>* scopes) -> bool {
  auto base_id = constant_values().GetInstId(base_const_id);
  auto base = insts().Get(base_id);
  if (auto base_as_namespace = base.TryAs<SemIR::Namespace>()) {
    scopes->push_back(
        LookupScope{.name_scope_id = base_as_namespace->name_scope_id,
                    .specific_id = SemIR::SpecificId::None});
    return true;
  }
  if (auto base_as_class = base.TryAs<SemIR::ClassType>()) {
    RequireDefinedType(
        *this, GetTypeIdForTypeConstant(base_const_id), loc_id, [&] {
          CARBON_DIAGNOSTIC(QualifiedExprInIncompleteClassScope, Error,
                            "member access into incomplete class {0}",
                            InstIdAsType);
          return emitter().Build(loc_id, QualifiedExprInIncompleteClassScope,
                                 base_id);
        });
    auto& class_info = classes().Get(base_as_class->class_id);
    scopes->push_back(LookupScope{.name_scope_id = class_info.scope_id,
                                  .specific_id = base_as_class->specific_id});
    return true;
  }
  if (auto base_as_facet_type = base.TryAs<SemIR::FacetType>()) {
    RequireDefinedType(
        *this, GetTypeIdForTypeConstant(base_const_id), loc_id, [&] {
          CARBON_DIAGNOSTIC(QualifiedExprInUndefinedInterfaceScope, Error,
                            "member access into undefined interface {0}",
                            InstIdAsType);
          return emitter().Build(loc_id, QualifiedExprInUndefinedInterfaceScope,
                                 base_id);
        });
    const auto& facet_type_info =
        facet_types().Get(base_as_facet_type->facet_type_id);
    for (auto interface : facet_type_info.impls_constraints) {
      auto& interface_info = interfaces().Get(interface.interface_id);
      scopes->push_back(LookupScope{.name_scope_id = interface_info.scope_id,
                                    .specific_id = interface.specific_id});
    }
    return true;
  }
  if (base_const_id == SemIR::ErrorInst::SingletonConstantId) {
    // Lookup into this scope should fail without producing an error.
    scopes->push_back(LookupScope{.name_scope_id = SemIR::NameScopeId::None,
                                  .specific_id = SemIR::SpecificId::None});
    return true;
  }
  // TODO: Per the design, if `base_id` is any kind of type, then lookup should
  // treat it as a name scope, even if it doesn't have members. For example,
  // `(i32*).X` should fail because there's no name `X` in `i32*`, not because
  // there's no name `X` in `type`.
  return false;
}

auto Context::LookupQualifiedName(SemIR::LocId loc_id, SemIR::NameId name_id,
                                  llvm::ArrayRef<LookupScope> lookup_scopes,
                                  bool required,
                                  std::optional<AccessInfo> access_info)
    -> LookupResult {
  llvm::SmallVector<LookupScope> scopes(lookup_scopes);

  // TODO: Support reporting of multiple prohibited access.
  llvm::SmallVector<ProhibitedAccessInfo> prohibited_accesses;

  LookupResult result = {
      .specific_id = SemIR::SpecificId::None,
      .scope_result = SemIR::ScopeLookupResult::MakeNotFound()};
  bool has_error = false;
  bool is_parent_access = false;

  // Walk this scope and, if nothing is found here, the scopes it extends.
  while (!scopes.empty()) {
    auto [scope_id, specific_id] = scopes.pop_back_val();
    if (!scope_id.has_value()) {
      has_error = true;
      continue;
    }
    auto& name_scope = name_scopes().Get(scope_id);
    has_error |= name_scope.has_error();

    const SemIR::ScopeLookupResult scope_result =
        LookupNameInExactScope(loc_id, name_id, scope_id, name_scope);
    SemIR::AccessKind access_kind = scope_result.access_kind();

    auto is_access_prohibited =
        IsAccessProhibited(access_info, access_kind, is_parent_access);

    // Keep track of prohibited accesses, this will be useful for reporting
    // multiple prohibited accesses if we can't find a suitable lookup.
    if (is_access_prohibited) {
      prohibited_accesses.push_back({
          .scope_result_id = scope_result.target_inst_id(),
          .access_kind = access_kind,
          .is_parent_access = is_parent_access,
      });
    }

    if (!scope_result.is_found() || is_access_prohibited) {
      // If nothing is found in this scope or if we encountered an invalid
      // access, look in its extended scopes.
      const auto& extended = name_scope.extended_scopes();
      scopes.reserve(scopes.size() + extended.size());
      for (auto extended_id : llvm::reverse(extended)) {
        // Substitute into the constant describing the extended scope to
        // determine its corresponding specific.
        CARBON_CHECK(extended_id.has_value());
        LoadImportRef(*this, extended_id);
        SemIR::ConstantId const_id =
            GetConstantValueInSpecific(sem_ir(), specific_id, extended_id);

        DiagnosticAnnotationScope annotate_diagnostics(
            &emitter(), [&](auto& builder) {
              CARBON_DIAGNOSTIC(FromExtendHere, Note,
                                "declared as an extended scope here");
              builder.Note(extended_id, FromExtendHere);
            });
        if (!AppendLookupScopesForConstant(loc_id, const_id, &scopes)) {
          // TODO: Handle case where we have a symbolic type and instead should
          // look in its type.
        }
      }
      is_parent_access |= !extended.empty();
      continue;
    }

    // If this is our second lookup result, diagnose an ambiguity.
    if (result.scope_result.is_found()) {
      CARBON_DIAGNOSTIC(
          NameAmbiguousDueToExtend, Error,
          "ambiguous use of name `{0}` found in multiple extended scopes",
          SemIR::NameId);
      emitter_->Emit(loc_id, NameAmbiguousDueToExtend, name_id);
      // TODO: Add notes pointing to the scopes.
      return {.specific_id = SemIR::SpecificId::None,
              .scope_result = SemIR::ScopeLookupResult::MakeError()};
    }

    result.scope_result = scope_result;
    result.specific_id = specific_id;
  }

  if (required && !result.scope_result.is_found()) {
    if (!has_error) {
      if (prohibited_accesses.empty()) {
        DiagnoseMemberNameNotFound(loc_id, name_id, lookup_scopes);
      } else {
        //  TODO: We should report multiple prohibited accesses in case we don't
        //  find a valid lookup. Reporting the last one should suffice for now.
        auto [scope_result_id, access_kind, is_parent_access] =
            prohibited_accesses.back();

        // Note, `access_info` is guaranteed to have a value here, since
        // `prohibited_accesses` is non-empty.
        DiagnoseInvalidQualifiedNameAccess(*this, loc_id, scope_result_id,
                                           name_id, access_kind,
                                           is_parent_access, *access_info);
      }
    }

    CARBON_CHECK(!result.scope_result.is_poisoned());
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeError()};
  }

  return result;
}

// Returns the scope of the Core package, or `None` if it's not found.
//
// TODO: Consider tracking the Core package in SemIR so we don't need to use
// name lookup to find it.
static auto GetCorePackage(Context& context, SemIR::LocId loc_id,
                           llvm::StringRef name) -> SemIR::NameScopeId {
  auto packaging = context.parse_tree().packaging_decl();
  if (packaging && packaging->names.package_id == PackageNameId::Core) {
    return SemIR::NameScopeId::Package;
  }
  auto core_name_id = SemIR::NameId::Core;

  // Look up `package.Core`.
  auto core_scope_result = context.LookupNameInExactScope(
      loc_id, core_name_id, SemIR::NameScopeId::Package,
      context.name_scopes().Get(SemIR::NameScopeId::Package));
  if (core_scope_result.is_found()) {
    // We expect it to be a namespace.
    if (auto namespace_inst = context.insts().TryGetAs<SemIR::Namespace>(
            core_scope_result.target_inst_id())) {
      // TODO: Decide whether to allow the case where `Core` is not a package.
      return namespace_inst->name_scope_id;
    }
  }

  CARBON_DIAGNOSTIC(
      CoreNotFound, Error,
      "`Core.{0}` implicitly referenced here, but package `Core` not found",
      std::string);
  context.emitter().Emit(loc_id, CoreNotFound, name.str());
  return SemIR::NameScopeId::None;
}

auto Context::LookupNameInCore(SemIR::LocId loc_id, llvm::StringRef name)
    -> SemIR::InstId {
  auto core_package_id = GetCorePackage(*this, loc_id, name);
  if (!core_package_id.has_value()) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  auto name_id = SemIR::NameId::ForIdentifier(identifiers().Add(name));
  auto scope_result = LookupNameInExactScope(
      loc_id, name_id, core_package_id, name_scopes().Get(core_package_id));
  if (!scope_result.is_found()) {
    CARBON_DIAGNOSTIC(
        CoreNameNotFound, Error,
        "name `Core.{0}` implicitly referenced here, but not found",
        SemIR::NameId);
    emitter_->Emit(loc_id, CoreNameNotFound, name_id);
    return SemIR::ErrorInst::SingletonInstId;
  }

  // Look through import_refs and aliases.
  return constant_values().GetConstantInstId(scope_result.target_inst_id());
}

auto Context::BeginSubpattern() -> void {
  inst_block_stack().Push();
  region_stack_.PushRegion(inst_block_stack().PeekOrAdd());
}

auto Context::EndSubpatternAsExpr(SemIR::InstId result_id)
    -> SemIR::ExprRegionId {
  if (region_stack_.PeekRegion().size() > 1) {
    // End the exit block with a branch to a successor block, whose contents
    // will be determined later.
    AddInst(SemIR::LocIdAndInst::NoLoc<SemIR::Branch>(
        {.target_id = inst_blocks().AddDefaultValue()}));
  } else {
    // This single-block region will be inserted as a SpliceBlock, so we don't
    // need control flow out of it.
  }
  auto block_id = inst_block_stack().Pop();
  CARBON_CHECK(block_id == region_stack_.PeekRegion().back());

  // TODO: Is it possible to validate that this region is genuinely
  // single-entry, single-exit?
  return sem_ir().expr_regions().Add(
      {.block_ids = region_stack_.PopRegion(), .result_id = result_id});
}

auto Context::EndSubpatternAsEmpty() -> void {
  auto block_id = inst_block_stack().Pop();
  CARBON_CHECK(block_id == region_stack_.PeekRegion().back());
  CARBON_CHECK(region_stack_.PeekRegion().size() == 1);
  CARBON_CHECK(inst_blocks().Get(block_id).empty());
  region_stack_.PopAndDiscardRegion();
}

auto Context::InsertHere(SemIR::ExprRegionId region_id) -> SemIR::InstId {
  auto region = sem_ir_->expr_regions().Get(region_id);
  auto loc_id = insts().GetLocId(region.result_id);
  auto exit_block = inst_blocks().Get(region.block_ids.back());
  if (region.block_ids.size() == 1) {
    // TODO: Is it possible to avoid leaving an "orphan" block in the IR in the
    // first two cases?
    if (exit_block.empty()) {
      return region.result_id;
    }
    if (exit_block.size() == 1) {
      inst_block_stack_.AddInstId(exit_block.front());
      return region.result_id;
    }
    return AddInst<SemIR::SpliceBlock>(
        loc_id, {.type_id = insts().Get(region.result_id).type_id(),
                 .block_id = region.block_ids.front(),
                 .result_id = region.result_id});
  }
  if (region_stack_.empty()) {
    TODO(loc_id,
         "Control flow expressions are currently only supported inside "
         "functions.");
    return SemIR::ErrorInst::SingletonInstId;
  }
  AddInst(SemIR::LocIdAndInst::NoLoc<SemIR::Branch>(
      {.target_id = region.block_ids.front()}));
  inst_block_stack_.Pop();
  // TODO: this will cumulatively cost O(MN) running time for M blocks
  // at the Nth level of the stack. Figure out how to do better.
  region_stack_.AddToRegion(region.block_ids);
  auto resume_with_block_id =
      insts().GetAs<SemIR::Branch>(exit_block.back()).target_id;
  CARBON_CHECK(inst_blocks().GetOrEmpty(resume_with_block_id).empty());
  inst_block_stack_.Push(resume_with_block_id);
  region_stack_.AddToRegion(resume_with_block_id, loc_id);
  return region.result_id;
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
