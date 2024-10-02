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
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/merge.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/builtin_inst_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/formatter.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

Context::Context(const Lex::TokenizedBuffer& tokens, DiagnosticEmitter& emitter,
                 const Parse::Tree& parse_tree,
                 llvm::function_ref<const Parse::TreeAndSubtrees&()>
                     get_parse_tree_and_subtrees,
                 SemIR::File& sem_ir, llvm::raw_ostream* vlog_stream)
    : tokens_(&tokens),
      emitter_(&emitter),
      parse_tree_(&parse_tree),
      get_parse_tree_and_subtrees_(get_parse_tree_and_subtrees),
      sem_ir_(&sem_ir),
      vlog_stream_(vlog_stream),
      node_stack_(parse_tree, vlog_stream),
      inst_block_stack_("inst_block_stack_", sem_ir, vlog_stream),
      pattern_block_stack_("pattern_block_stack_", sem_ir, vlog_stream),
      param_and_arg_refs_stack_(sem_ir, vlog_stream, node_stack_),
      args_type_info_stack_("args_type_info_stack_", sem_ir, vlog_stream),
      decl_name_stack_(this),
      scope_stack_(sem_ir_->identifiers()),
      global_init_(this) {
  // Map the builtin `<error>` and `type` type constants to their corresponding
  // special `TypeId` values.
  type_ids_for_type_constants_.Insert(
      SemIR::ConstantId::ForTemplateConstant(SemIR::InstId::BuiltinError),
      SemIR::TypeId::Error);
  type_ids_for_type_constants_.Insert(
      SemIR::ConstantId::ForTemplateConstant(SemIR::InstId::BuiltinTypeType),
      SemIR::TypeId::TypeType);

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
  scope_stack_.VerifyOnFinish();
  inst_block_stack_.VerifyOnFinish();
  pattern_block_stack_.VerifyOnFinish();
  param_and_arg_refs_stack_.VerifyOnFinish();
}

// Finish producing an instruction. Set its constant value, and register it in
// any applicable instruction lists.
auto Context::FinishInst(SemIR::InstId inst_id, SemIR::Inst inst) -> void {
  GenericRegionStack::DependencyKind dep_kind =
      GenericRegionStack::DependencyKind::None;

  // If the instruction has a symbolic constant type, track that we need to
  // substitute into it.
  if (types().GetConstantId(inst.type_id()).is_symbolic()) {
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
    if (const_id.is_symbolic()) {
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

auto Context::AddInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG("AddInst: {0}\n", loc_id_and_inst.inst);
  FinishInst(inst_id, loc_id_and_inst.inst);
  return inst_id;
}

auto Context::AddInst(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(loc_id_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddPlaceholderInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG("AddPlaceholderInst: {0}\n", loc_id_and_inst.inst);
  constant_values().Set(inst_id, SemIR::ConstantId::Invalid);
  return inst_id;
}

auto Context::AddPlaceholderInst(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddPlaceholderInstInNoBlock(loc_id_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddPatternInst(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(loc_id_and_inst);
  pattern_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddConstant(SemIR::Inst inst, bool is_symbolic)
    -> SemIR::ConstantId {
  auto const_id = constants().GetOrAdd(inst, is_symbolic);
  CARBON_VLOG("AddConstant: {0}\n", inst);
  return const_id;
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

auto Context::DiagnoseDuplicateName(SemIRLoc dup_def, SemIRLoc prev_def)
    -> void {
  CARBON_DIAGNOSTIC(NameDeclDuplicate, Error,
                    "duplicate name being declared in the same scope");
  CARBON_DIAGNOSTIC(NameDeclPrevious, Note, "name is previously declared here");
  emitter_->Build(dup_def, NameDeclDuplicate)
      .Note(prev_def, NameDeclPrevious)
      .Emit();
}

auto Context::DiagnoseNameNotFound(SemIRLoc loc, SemIR::NameId name_id)
    -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "name `{0}` not found", SemIR::NameId);
  emitter_->Emit(loc, NameNotFound, name_id);
}

auto Context::NoteIncompleteClass(SemIR::ClassId class_id,
                                  DiagnosticBuilder& builder) -> void {
  const auto& class_info = classes().Get(class_id);
  CARBON_CHECK(!class_info.is_defined(), "Class is not incomplete");
  if (class_info.definition_id.is_valid()) {
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

auto Context::AddNameToLookup(SemIR::NameId name_id, SemIR::InstId target_id)
    -> void {
  if (auto existing = scope_stack().LookupOrAddName(name_id, target_id);
      existing.is_valid()) {
    DiagnoseDuplicateName(target_id, existing);
  }
}

auto Context::LookupNameInDecl(SemIR::LocId loc_id, SemIR::NameId name_id,
                               SemIR::NameScopeId scope_id) -> SemIR::InstId {
  if (!scope_id.is_valid()) {
    // Look for a name in the current scope only. There are two cases where the
    // name would be in an outer scope:
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
    return scope_stack().LookupInCurrentScope(name_id);
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
                                  name_scopes().Get(scope_id))
        .first;
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
    if (auto non_lexical_result = LookupQualifiedName(
            node_id, name_id,
            {.name_scope_id = lookup_scope_id, .specific_id = specific_id},
            /*required=*/false);
        non_lexical_result.inst_id.is_valid()) {
      return non_lexical_result;
    }
  }

  if (lexical_result.is_valid()) {
    // A lexical scope never needs an associated specific. If there's a
    // lexically enclosing generic, then it also encloses the point of use of
    // the name.
    return {.specific_id = SemIR::SpecificId::Invalid,
            .inst_id = lexical_result};
  }

  // We didn't find anything at all.
  if (required) {
    DiagnoseNameNotFound(node_id, name_id);
  }

  return {.specific_id = SemIR::SpecificId::Invalid,
          .inst_id = SemIR::InstId::BuiltinError};
}

auto Context::LookupNameInExactScope(SemIRLoc loc, SemIR::NameId name_id,
                                     SemIR::NameScopeId scope_id,
                                     const SemIR::NameScope& scope)
    -> std::pair<SemIR::InstId, SemIR::AccessKind> {
  if (auto lookup = scope.name_map.Lookup(name_id)) {
    auto entry = scope.names[lookup.value()];
    LoadImportRef(*this, entry.inst_id);
    return {entry.inst_id, entry.access_kind};
  }

  if (!scope.import_ir_scopes.empty()) {
    // TODO: Enforce other access modifiers for imports.
    return {ImportNameFromOtherPackage(*this, loc, scope_id,
                                       scope.import_ir_scopes, name_id),
            SemIR::AccessKind::Public};
  }
  return {SemIR::InstId::Invalid, SemIR::AccessKind::Public};
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
  auto class_info = context.classes().Get(class_type->class_id);

  CARBON_DIAGNOSTIC(ClassInvalidMemberAccess, Error,
                    "cannot access {0} member `{1}` of type `{2}`",
                    SemIR::AccessKind, SemIR::NameId, SemIR::TypeId);
  CARBON_DIAGNOSTIC(ClassMemberDefinition, Note,
                    "the {0} member `{1}` is defined here", SemIR::AccessKind,
                    SemIR::NameId);

  auto parent_type_id = class_info.self_type_id;

  if (access_kind == SemIR::AccessKind::Private && is_parent_access) {
    if (auto base_decl = context.insts().TryGetAsIfValid<SemIR::BaseDecl>(
            class_info.base_id)) {
      parent_type_id = base_decl->base_type_id;
    } else if (auto adapt_decl =
                   context.insts().TryGetAsIfValid<SemIR::AdaptDecl>(
                       class_info.adapt_id)) {
      parent_type_id = adapt_decl->adapted_type_id;
    } else {
      CARBON_FATAL("Expected parent for parent access");
    }
  }

  context.emitter()
      .Build(loc, ClassInvalidMemberAccess, access_kind, name_id,
             parent_type_id)
      .Note(scope_result_id, ClassMemberDefinition, access_kind, name_id)
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

auto Context::LookupQualifiedName(SemIRLoc loc, SemIR::NameId name_id,
                                  LookupScope scope, bool required,
                                  std::optional<AccessInfo> access_info)
    -> LookupResult {
  llvm::SmallVector<LookupScope> scopes = {scope};

  // TODO: Support reporting of multiple prohibited access.
  llvm::SmallVector<ProhibitedAccessInfo> prohibited_accesses;

  LookupResult result = {.specific_id = SemIR::SpecificId::Invalid,
                         .inst_id = SemIR::InstId::Invalid};
  bool has_error = false;
  bool is_parent_access = false;

  // Walk this scope and, if nothing is found here, the scopes it extends.
  while (!scopes.empty()) {
    auto [scope_id, specific_id] = scopes.pop_back_val();
    const auto& name_scope = name_scopes().Get(scope_id);
    has_error |= name_scope.has_error;

    auto [scope_result_id, access_kind] =
        LookupNameInExactScope(loc, name_id, scope_id, name_scope);

    auto is_access_prohibited =
        IsAccessProhibited(access_info, access_kind, is_parent_access);

    // Keep track of prohibited accesses, this will be useful for reporting
    // multiple prohibited accesses if we can't find a suitable lookup.
    if (is_access_prohibited) {
      prohibited_accesses.push_back({
          .scope_result_id = scope_result_id,
          .access_kind = access_kind,
          .is_parent_access = is_parent_access,
      });
    }

    if (!scope_result_id.is_valid() || is_access_prohibited) {
      // If nothing is found in this scope or if we encountered an invalid
      // access, look in its extended scopes.
      auto extended = name_scope.extended_scopes;
      scopes.reserve(scopes.size() + extended.size());
      for (auto extended_id : llvm::reverse(extended)) {
        // TODO: Track a constant describing the extended scope, and substitute
        // into it to determine its corresponding specific.
        scopes.push_back({.name_scope_id = extended_id,
                          .specific_id = SemIR::SpecificId::Invalid});
      }
      is_parent_access |= !extended.empty();
      continue;
    }

    // If this is our second lookup result, diagnose an ambiguity.
    if (result.inst_id.is_valid()) {
      // TODO: This is currently not reachable because the only scope that can
      // extend is a class scope, and it can only extend a single base class.
      // Add test coverage once this is possible.
      CARBON_DIAGNOSTIC(
          NameAmbiguousDueToExtend, Error,
          "ambiguous use of name `{0}` found in multiple extended scopes",
          SemIR::NameId);
      emitter_->Emit(loc, NameAmbiguousDueToExtend, name_id);
      // TODO: Add notes pointing to the scopes.
      return {.specific_id = SemIR::SpecificId::Invalid,
              .inst_id = SemIR::InstId::BuiltinError};
    }

    result.inst_id = scope_result_id;
    result.specific_id = specific_id;
  }

  if (required && !result.inst_id.is_valid()) {
    if (!has_error) {
      if (prohibited_accesses.empty()) {
        DiagnoseNameNotFound(loc, name_id);
      } else {
        //  TODO: We should report multiple prohibited accesses in case we don't
        //  find a valid lookup. Reporting the last one should suffice for now.
        auto [scope_result_id, access_kind, is_parent_access] =
            prohibited_accesses.back();

        // Note, `access_info` is guaranteed to have a value here, since
        // `prohibited_accesses` is non-empty.
        DiagnoseInvalidQualifiedNameAccess(*this, loc, scope_result_id, name_id,
                                           access_kind, is_parent_access,
                                           *access_info);
      }
    }

    return {.specific_id = SemIR::SpecificId::Invalid,
            .inst_id = SemIR::InstId::BuiltinError};
  }

  return result;
}

// Returns the scope of the Core package, or Invalid if it's not found.
//
// TODO: Consider tracking the Core package in SemIR so we don't need to use
// name lookup to find it.
static auto GetCorePackage(Context& context, SemIRLoc loc)
    -> SemIR::NameScopeId {
  auto core_ident_id = context.identifiers().Add("Core");
  auto packaging = context.parse_tree().packaging_decl();
  if (packaging && packaging->names.package_id == core_ident_id) {
    return SemIR::NameScopeId::Package;
  }
  auto core_name_id = SemIR::NameId::ForIdentifier(core_ident_id);

  // Look up `package.Core`.
  auto [core_inst_id, _] = context.LookupNameInExactScope(
      loc, core_name_id, SemIR::NameScopeId::Package,
      context.name_scopes().Get(SemIR::NameScopeId::Package));
  if (core_inst_id.is_valid()) {
    // We expect it to be a namespace.
    if (auto namespace_inst =
            context.insts().TryGetAs<SemIR::Namespace>(core_inst_id)) {
      // TODO: Decide whether to allow the case where `Core` is not a package.
      return namespace_inst->name_scope_id;
    }
  }

  CARBON_DIAGNOSTIC(CoreNotFound, Error,
                    "package `Core` implicitly referenced here, but not found");
  context.emitter().Emit(loc, CoreNotFound);
  return SemIR::NameScopeId::Invalid;
}

auto Context::LookupNameInCore(SemIRLoc loc, llvm::StringRef name)
    -> SemIR::InstId {
  auto core_package_id = GetCorePackage(*this, loc);
  if (!core_package_id.is_valid()) {
    return SemIR::InstId::BuiltinError;
  }

  auto name_id = SemIR::NameId::ForIdentifier(identifiers().Add(name));
  auto [inst_id, _] = LookupNameInExactScope(
      loc, name_id, core_package_id, name_scopes().Get(core_package_id));
  if (!inst_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        CoreNameNotFound, Error,
        "name `Core.{0}` implicitly referenced here, but not found",
        SemIR::NameId);
    emitter_->Emit(loc, CoreNameNotFound, name_id);
    return SemIR::InstId::BuiltinError;
  }

  // Look through import_refs and aliases.
  return constant_values().GetConstantInstId(inst_id);
}

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::NodeId node_id, Args... args)
    -> SemIR::InstBlockId {
  if (!context.inst_block_stack().is_current_block_reachable()) {
    return SemIR::InstBlockId::Unreachable;
  }
  auto block_id = context.inst_blocks().AddDefaultValue();
  context.AddInst<BranchNode>(node_id, {block_id, args...});
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::NodeId node_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Branch>(*this, node_id);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::NodeId node_id,
                                                SemIR::InstId arg_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchWithArg>(*this, node_id,
                                                              arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::NodeId node_id,
                                           SemIR::InstId cond_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchIf>(*this, node_id,
                                                         cond_id);
}

auto Context::AddConvergenceBlockAndPush(Parse::NodeId node_id, int num_blocks)
    -> void {
  CARBON_CHECK(num_blocks >= 2, "no convergence");

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst<SemIR::Branch>(node_id, {.target_id = new_block_id});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::NodeId node_id, std::initializer_list<SemIR::InstId> block_args)
    -> SemIR::InstId {
  CARBON_CHECK(block_args.size() >= 2, "no convergence");

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst<SemIR::BranchWithArg>(
          node_id, {.target_id = new_block_id, .arg_id = arg_id});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id = insts().Get(*block_args.begin()).type_id();
  return AddInst<SemIR::BlockArg>(
      node_id, {.type_id = result_type_id, .block_id = new_block_id});
}

auto Context::SetBlockArgResultBeforeConstantUse(SemIR::InstId select_id,
                                                 SemIR::InstId cond_id,
                                                 SemIR::InstId if_true,
                                                 SemIR::InstId if_false)
    -> void {
  CARBON_CHECK(insts().Is<SemIR::BlockArg>(select_id));

  // Determine the constant result based on the condition value.
  SemIR::ConstantId const_id = SemIR::ConstantId::NotConstant;
  auto cond_const_id = constant_values().Get(cond_id);
  if (!cond_const_id.is_template()) {
    // Symbolic or non-constant condition means a non-constant result.
  } else if (auto literal = insts().TryGetAs<SemIR::BoolLiteral>(
                 constant_values().GetInstId(cond_const_id))) {
    const_id = constant_values().Get(literal.value().value.ToBool() ? if_true
                                                                    : if_false);
  } else {
    CARBON_CHECK(cond_const_id == SemIR::ConstantId::Error,
                 "Unexpected constant branch condition.");
    const_id = SemIR::ConstantId::Error;
  }

  if (const_id.is_constant()) {
    CARBON_VLOG("Constant: {0} -> {1}\n", insts().Get(select_id),
                constant_values().GetInstId(const_id));
    constant_values().Set(select_id, const_id);
  }
}

auto Context::AddCurrentCodeBlockToFunction(Parse::NodeId node_id) -> void {
  CARBON_CHECK(!inst_block_stack().empty(), "no current code block");

  if (return_scope_stack().empty()) {
    CARBON_CHECK(node_id.is_valid(),
                 "No current function, but node_id not provided");
    TODO(node_id,
         "Control flow expressions are currently only supported inside "
         "functions.");
    return;
  }

  if (!inst_block_stack().is_current_block_reachable()) {
    // Don't include unreachable blocks in the function.
    return;
  }

  auto function_id =
      insts()
          .GetAs<SemIR::FunctionDecl>(return_scope_stack().back().decl_id)
          .function_id;
  functions()
      .Get(function_id)
      .body_block_ids.push_back(inst_block_stack().PeekOrAdd());
}

auto Context::is_current_position_reachable() -> bool {
  if (!inst_block_stack().is_current_block_reachable()) {
    return false;
  }

  // Our current position is at the end of a reachable block. That position is
  // reachable unless the previous instruction is a terminator instruction.
  auto block_contents = inst_block_stack().PeekCurrentBlockContents();
  if (block_contents.empty()) {
    return true;
  }
  const auto& last_inst = insts().Get(block_contents.back());
  return last_inst.kind().terminator_kind() !=
         SemIR::TerminatorKind::Terminator;
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

namespace {
// Worklist-based type completion mechanism.
//
// When attempting to complete a type, we may find other types that also need to
// be completed: types nested within that type, and the value representation of
// the type. In order to complete a type without recursing arbitrarily deeply,
// we use a worklist of tasks:
//
// - An `AddNestedIncompleteTypes` step adds a task for all incomplete types
//   nested within a type to the work list.
// - A `BuildValueRepr` step computes the value representation for a
//   type, once all of its nested types are complete, and marks the type as
//   complete.
class TypeCompleter {
 public:
  TypeCompleter(Context& context,
                std::optional<Context::BuildDiagnosticFn> diagnoser)
      : context_(context), diagnoser_(diagnoser) {}

  // Attempts to complete the given type. Returns true if it is now complete,
  // false if it could not be completed.
  auto Complete(SemIR::TypeId type_id) -> bool {
    Push(type_id);
    while (!work_list_.empty()) {
      if (!ProcessStep()) {
        return false;
      }
    }
    return true;
  }

 private:
  // Adds `type_id` to the work list, if it's not already complete.
  auto Push(SemIR::TypeId type_id) -> void {
    if (!context_.types().IsComplete(type_id)) {
      work_list_.push_back(
          {.type_id = type_id, .phase = Phase::AddNestedIncompleteTypes});
    }
  }

  // Runs the next step.
  auto ProcessStep() -> bool {
    auto [type_id, phase] = work_list_.back();

    // We might have enqueued the same type more than once. Just skip the
    // type if it's already complete.
    if (context_.types().IsComplete(type_id)) {
      work_list_.pop_back();
      return true;
    }

    auto inst_id = context_.types().GetInstId(type_id);
    auto inst = context_.insts().Get(inst_id);
    auto old_work_list_size = work_list_.size();

    switch (phase) {
      case Phase::AddNestedIncompleteTypes:
        if (!AddNestedIncompleteTypes(inst)) {
          return false;
        }
        CARBON_CHECK(work_list_.size() >= old_work_list_size,
                     "AddNestedIncompleteTypes should not remove work items");
        work_list_[old_work_list_size - 1].phase = Phase::BuildValueRepr;
        break;

      case Phase::BuildValueRepr: {
        auto value_rep = BuildValueRepr(type_id, inst);
        context_.types().SetValueRepr(type_id, value_rep);
        CARBON_CHECK(old_work_list_size == work_list_.size(),
                     "BuildValueRepr should not change work items");
        work_list_.pop_back();

        // Also complete the value representation type, if necessary. This
        // should never fail: the value representation shouldn't require any
        // additional nested types to be complete.
        if (!context_.types().IsComplete(value_rep.type_id)) {
          work_list_.push_back(
              {.type_id = value_rep.type_id, .phase = Phase::BuildValueRepr});
        }
        // For a pointer representation, the pointee also needs to be complete.
        if (value_rep.kind == SemIR::ValueRepr::Pointer) {
          if (value_rep.type_id == SemIR::TypeId::Error) {
            break;
          }
          auto pointee_type_id =
              context_.sem_ir().GetPointeeType(value_rep.type_id);
          if (!context_.types().IsComplete(pointee_type_id)) {
            work_list_.push_back(
                {.type_id = pointee_type_id, .phase = Phase::BuildValueRepr});
          }
        }
        break;
      }
    }

    return true;
  }

  // Adds any types nested within `type_inst` that need to be complete for
  // `type_inst` to be complete to our work list.
  auto AddNestedIncompleteTypes(SemIR::Inst type_inst) -> bool {
    CARBON_KIND_SWITCH(type_inst) {
      case CARBON_KIND(SemIR::ArrayType inst): {
        Push(inst.element_type_id);
        break;
      }
      case CARBON_KIND(SemIR::StructType inst): {
        for (auto field_id : context_.inst_blocks().Get(inst.fields_id)) {
          Push(context_.insts()
                   .GetAs<SemIR::StructTypeField>(field_id)
                   .field_type_id);
        }
        break;
      }
      case CARBON_KIND(SemIR::TupleType inst): {
        for (auto element_type_id :
             context_.type_blocks().Get(inst.elements_id)) {
          Push(element_type_id);
        }
        break;
      }
      case CARBON_KIND(SemIR::ClassType inst): {
        auto& class_info = context_.classes().Get(inst.class_id);
        if (!class_info.is_defined()) {
          if (diagnoser_) {
            auto builder = (*diagnoser_)();
            context_.NoteIncompleteClass(inst.class_id, builder);
            builder.Emit();
          }
          return false;
        }
        if (inst.specific_id.is_valid()) {
          ResolveSpecificDefinition(context_, inst.specific_id);
        }
        Push(class_info.GetObjectRepr(context_.sem_ir(), inst.specific_id));
        break;
      }
      case CARBON_KIND(SemIR::ConstType inst): {
        Push(inst.inner_id);
        break;
      }
      default:
        break;
    }

    return true;
  }

  // Makes an empty value representation, which is used for types that have no
  // state, such as empty structs and tuples.
  auto MakeEmptyValueRepr() const -> SemIR::ValueRepr {
    return {.kind = SemIR::ValueRepr::None,
            .type_id = context_.GetTupleType({})};
  }

  // Makes a value representation that uses pass-by-copy, copying the given
  // type.
  auto MakeCopyValueRepr(SemIR::TypeId rep_id,
                         SemIR::ValueRepr::AggregateKind aggregate_kind =
                             SemIR::ValueRepr::NotAggregate) const
      -> SemIR::ValueRepr {
    return {.kind = SemIR::ValueRepr::Copy,
            .aggregate_kind = aggregate_kind,
            .type_id = rep_id};
  }

  // Makes a value representation that uses pass-by-address with the given
  // pointee type.
  auto MakePointerValueRepr(SemIR::TypeId pointee_id,
                            SemIR::ValueRepr::AggregateKind aggregate_kind =
                                SemIR::ValueRepr::NotAggregate) const
      -> SemIR::ValueRepr {
    // TODO: Should we add `const` qualification to `pointee_id`?
    return {.kind = SemIR::ValueRepr::Pointer,
            .aggregate_kind = aggregate_kind,
            .type_id = context_.GetPointerType(pointee_id)};
  }

  // Gets the value representation of a nested type, which should already be
  // complete.
  auto GetNestedValueRepr(SemIR::TypeId nested_type_id) const {
    CARBON_CHECK(context_.types().IsComplete(nested_type_id),
                 "Nested type should already be complete");
    auto value_rep = context_.types().GetValueRepr(nested_type_id);
    CARBON_CHECK(value_rep.kind != SemIR::ValueRepr::Unknown,
                 "Complete type should have a value representation");
    return value_rep;
  }

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::BuiltinInst builtin) const
      -> SemIR::ValueRepr {
    switch (builtin.builtin_inst_kind) {
      case SemIR::BuiltinInstKind::TypeType:
      case SemIR::BuiltinInstKind::Error:
      case SemIR::BuiltinInstKind::Invalid:
      case SemIR::BuiltinInstKind::BoolType:
      case SemIR::BuiltinInstKind::IntType:
      case SemIR::BuiltinInstKind::FloatType:
      case SemIR::BuiltinInstKind::NamespaceType:
      case SemIR::BuiltinInstKind::BoundMethodType:
      case SemIR::BuiltinInstKind::WitnessType:
        return MakeCopyValueRepr(type_id);

      case SemIR::BuiltinInstKind::StringType:
        // TODO: Decide on string value semantics. This should probably be a
        // custom value representation carrying a pointer and size or
        // similar.
        return MakePointerValueRepr(type_id);
    }
    llvm_unreachable("All builtin kinds were handled above");
  }

  auto BuildStructOrTupleValueRepr(std::size_t num_elements,
                                   SemIR::TypeId elementwise_rep,
                                   bool same_as_object_rep) const
      -> SemIR::ValueRepr {
    SemIR::ValueRepr::AggregateKind aggregate_kind =
        same_as_object_rep ? SemIR::ValueRepr::ValueAndObjectAggregate
                           : SemIR::ValueRepr::ValueAggregate;

    if (num_elements == 1) {
      // The value representation for a struct or tuple with a single element
      // is a struct or tuple containing the value representation of the
      // element.
      // TODO: Consider doing the same whenever `elementwise_rep` is
      // sufficiently small.
      return MakeCopyValueRepr(elementwise_rep, aggregate_kind);
    }
    // For a struct or tuple with multiple fields, we use a pointer
    // to the elementwise value representation.
    return MakePointerValueRepr(elementwise_rep, aggregate_kind);
  }

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::StructType struct_type) const
      -> SemIR::ValueRepr {
    // TODO: Share more code with tuples.
    auto fields = context_.inst_blocks().Get(struct_type.fields_id);
    if (fields.empty()) {
      return MakeEmptyValueRepr();
    }

    // Find the value representation for each field, and construct a struct
    // of value representations.
    llvm::SmallVector<SemIR::InstId> value_rep_fields;
    value_rep_fields.reserve(fields.size());
    bool same_as_object_rep = true;
    for (auto field_id : fields) {
      auto field = context_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto field_value_rep = GetNestedValueRepr(field.field_type_id);
      if (field_value_rep.type_id != field.field_type_id) {
        same_as_object_rep = false;
        field.field_type_id = field_value_rep.type_id;
        field_id = context_.constant_values().GetInstId(
            TryEvalInst(context_, SemIR::InstId::Invalid, field));
      }
      value_rep_fields.push_back(field_id);
    }

    auto value_rep = same_as_object_rep
                         ? type_id
                         : context_.GetStructType(
                               context_.inst_blocks().Add(value_rep_fields));
    return BuildStructOrTupleValueRepr(fields.size(), value_rep,
                                       same_as_object_rep);
  }

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::TupleType tuple_type) const
      -> SemIR::ValueRepr {
    // TODO: Share more code with structs.
    auto elements = context_.type_blocks().Get(tuple_type.elements_id);
    if (elements.empty()) {
      return MakeEmptyValueRepr();
    }

    // Find the value representation for each element, and construct a tuple
    // of value representations.
    llvm::SmallVector<SemIR::TypeId> value_rep_elements;
    value_rep_elements.reserve(elements.size());
    bool same_as_object_rep = true;
    for (auto element_type_id : elements) {
      auto element_value_rep = GetNestedValueRepr(element_type_id);
      if (element_value_rep.type_id != element_type_id) {
        same_as_object_rep = false;
      }
      value_rep_elements.push_back(element_value_rep.type_id);
    }

    auto value_rep = same_as_object_rep
                         ? type_id
                         : context_.GetTupleType(value_rep_elements);
    return BuildStructOrTupleValueRepr(elements.size(), value_rep,
                                       same_as_object_rep);
  }

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::ArrayType /*inst*/) const
      -> SemIR::ValueRepr {
    // For arrays, it's convenient to always use a pointer representation,
    // even when the array has zero or one element, in order to support
    // indexing.
    return MakePointerValueRepr(type_id, SemIR::ValueRepr::ObjectAggregate);
  }

  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/,
                             SemIR::ClassType inst) const -> SemIR::ValueRepr {
    auto& class_info = context_.classes().Get(inst.class_id);
    // The value representation of an adapter is the value representation of
    // its adapted type.
    if (class_info.adapt_id.is_valid()) {
      return GetNestedValueRepr(SemIR::GetTypeInSpecific(
          context_.sem_ir(), inst.specific_id,
          context_.insts()
              .GetAs<SemIR::AdaptDecl>(class_info.adapt_id)
              .adapted_type_id));
    }
    // Otherwise, the value representation for a class is a pointer to the
    // object representation.
    // TODO: Support customized value representations for classes.
    // TODO: Pick a better value representation when possible.
    return MakePointerValueRepr(
        class_info.GetObjectRepr(context_.sem_ir(), inst.specific_id),
        SemIR::ValueRepr::ObjectAggregate);
  }

  template <typename InstT>
    requires(
        InstT::Kind
            .template IsAnyOf<SemIR::AssociatedEntityType, SemIR::FunctionType,
                              SemIR::GenericClassType,
                              SemIR::GenericInterfaceType, SemIR::InterfaceType,
                              SemIR::UnboundElementType, SemIR::WhereExpr>())
  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/, InstT /*inst*/) const
      -> SemIR::ValueRepr {
    // These types have no runtime operations, so we use an empty value
    // representation.
    //
    // TODO: There is information we could model here:
    // - For an interface, we could use a witness.
    // - For an associated entity, we could use an index into the witness.
    // - For an unbound element, we could use an index or offset.
    return MakeEmptyValueRepr();
  }

  template <typename InstT>
    requires(InstT::Kind.template IsAnyOf<SemIR::BindSymbolicName,
                                          SemIR::InterfaceWitnessAccess>())
  auto BuildValueReprForInst(SemIR::TypeId type_id, InstT /*inst*/) const
      -> SemIR::ValueRepr {
    // For symbolic types, we arbitrarily pick a copy representation.
    return MakeCopyValueRepr(type_id);
  }

  template <typename InstT>
    requires(InstT::Kind.template IsAnyOf<SemIR::FloatType, SemIR::IntType,
                                          SemIR::PointerType>())
  auto BuildValueReprForInst(SemIR::TypeId type_id, InstT /*inst*/) const
      -> SemIR::ValueRepr {
    return MakeCopyValueRepr(type_id);
  }

  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/,
                             SemIR::ConstType inst) const -> SemIR::ValueRepr {
    // The value representation of `const T` is the same as that of `T`.
    // Objects are not modifiable through their value representations.
    return GetNestedValueRepr(inst.inner_id);
  }

  template <typename InstT>
    requires(InstT::Kind.is_type() == SemIR::InstIsType::Never)
  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/, InstT inst) const
      -> SemIR::ValueRepr {
    CARBON_FATAL("Type refers to non-type inst {0}", inst);
  }

  // Builds and returns the value representation for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildValueRepr(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::ValueRepr {
    // Use overload resolution to select the implementation, producing compile
    // errors when BuildTypeForInst isn't defined for a given instruction.
    CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(Name)                  \
  case CARBON_KIND(SemIR::Name typed_inst): {          \
    return BuildValueReprForInst(type_id, typed_inst); \
  }
#include "toolchain/sem_ir/inst_kind.def"
    }
  }

  enum class Phase : int8_t {
    // The next step is to add nested types to the list of types to complete.
    AddNestedIncompleteTypes,
    // The next step is to build the value representation for the type.
    BuildValueRepr,
  };

  struct WorkItem {
    SemIR::TypeId type_id;
    Phase phase;
  };

  Context& context_;
  llvm::SmallVector<WorkItem> work_list_;
  std::optional<Context::BuildDiagnosticFn> diagnoser_;
};
}  // namespace

auto Context::TryToCompleteType(SemIR::TypeId type_id,
                                std::optional<BuildDiagnosticFn> diagnoser)
    -> bool {
  return TypeCompleter(*this, diagnoser).Complete(type_id);
}

auto Context::TryToDefineType(SemIR::TypeId type_id,
                              std::optional<BuildDiagnosticFn> diagnoser)
    -> bool {
  if (!TryToCompleteType(type_id, diagnoser)) {
    return false;
  }

  if (auto interface = types().TryGetAs<SemIR::InterfaceType>(type_id)) {
    auto interface_id = interface->interface_id;
    if (!interfaces().Get(interface_id).is_defined()) {
      auto builder = (*diagnoser)();
      NoteUndefinedInterface(interface_id, builder);
      builder.Emit();
      return false;
    }

    if (interface->specific_id.is_valid()) {
      ResolveSpecificDefinition(*this, interface->specific_id);
    }
  }

  return true;
}

auto Context::GetTypeIdForTypeConstant(SemIR::ConstantId constant_id)
    -> SemIR::TypeId {
  CARBON_CHECK(constant_id.is_constant(),
               "Canonicalizing non-constant type: {0}", constant_id);
  auto type_id =
      insts().Get(constant_values().GetInstId(constant_id)).type_id();
  // TODO: For now, we allow values of facet type to be used as types.
  CARBON_CHECK(IsFacetType(type_id) || constant_id == SemIR::ConstantId::Error,
               "Forming type ID for non-type constant of type {0}",
               types().GetAsInst(type_id));

  return SemIR::TypeId::ForTypeConstant(constant_id);
}

// Gets or forms a type_id for a type, given the instruction kind and arguments.
template <typename InstT, typename... EachArgT>
static auto GetTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  // TODO: Remove inst_id parameter from TryEvalInst.
  InstT inst = {SemIR::TypeId::TypeType, each_arg...};
  return context.GetTypeIdForTypeConstant(
      TryEvalInst(context, SemIR::InstId::Invalid, inst));
}

// Gets or forms a type_id for a type, given the instruction kind and arguments,
// and completes the type. This should only be used when type completion cannot
// fail.
template <typename InstT, typename... EachArgT>
static auto GetCompleteTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  auto type_id = GetTypeImpl<InstT>(context, each_arg...);
  bool complete = context.TryToCompleteType(type_id);
  CARBON_CHECK(complete, "Type completion should not fail");
  return type_id;
}

auto Context::GetStructType(SemIR::InstBlockId refs_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::StructType>(*this, refs_id);
}

auto Context::GetTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::TupleType>(*this,
                                       type_blocks().AddCanonical(type_ids));
}

auto Context::GetAssociatedEntityType(SemIR::TypeId interface_type_id,
                                      SemIR::TypeId entity_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::AssociatedEntityType>(*this, interface_type_id,
                                                  entity_type_id);
}

auto Context::GetBuiltinType(SemIR::BuiltinInstKind kind) -> SemIR::TypeId {
  CARBON_CHECK(kind != SemIR::BuiltinInstKind::Invalid);
  auto type_id = GetTypeIdForTypeInst(SemIR::InstId::ForBuiltin(kind));
  // To keep client code simpler, complete builtin types before returning them.
  bool complete = TryToCompleteType(type_id);
  CARBON_CHECK(complete, "Failed to complete builtin type");
  return type_id;
}

auto Context::GetFunctionType(SemIR::FunctionId fn_id,
                              SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::FunctionType>(*this, fn_id, specific_id);
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

auto Context::GetPointerType(SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::PointerType>(*this, pointee_type_id);
}

auto Context::GetUnboundElementType(SemIR::TypeId class_type_id,
                                    SemIR::TypeId element_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::UnboundElementType>(*this, class_type_id,
                                                element_type_id);
}

auto Context::GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId {
  if (auto const_type = types().TryGetAs<SemIR::ConstType>(type_id)) {
    return const_type->inner_id;
  }
  return type_id;
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  output << "Check::Context\n";

  // In a stack dump, this is probably indented by a tab. We treat that as 8
  // spaces then add a couple to indent past the Context label.
  constexpr int Indent = 10;

  SemIR::Formatter formatter(*tokens_, *parse_tree_, *sem_ir_);
  node_stack_.PrintForStackDump(formatter, Indent, output);
  inst_block_stack_.PrintForStackDump(formatter, Indent, output);
  pattern_block_stack_.PrintForStackDump(formatter, Indent, output);
  param_and_arg_refs_stack_.PrintForStackDump(formatter, Indent, output);
  args_type_info_stack_.PrintForStackDump(formatter, Indent, output);
}

auto Context::DumpFormattedFile() const -> void {
  SemIR::Formatter formatter(*tokens_, *parse_tree_, *sem_ir_);
  formatter.Print(llvm::errs());
}

}  // namespace Carbon::Check
