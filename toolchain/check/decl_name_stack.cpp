// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/decl_name_stack.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

auto DeclNameStack::NameContext::prev_inst_id() const -> SemIR::InstId {
  switch (state) {
    case NameContext::State::Error:
      // The name is malformed and a diagnostic has already been emitted.
      return SemIR::InstId::None;

    case NameContext::State::Empty:
      CARBON_FATAL(
          "Name is missing, not expected to call existing_inst_id (but that "
          "may change based on error handling).");

    case NameContext::State::Resolved:
      return resolved_inst_id;

    case NameContext::State::Unresolved:
      return SemIR::InstId::None;

    case NameContext::State::Poisoned:
      CARBON_FATAL("Poisoned state should not call prev_inst_id()");

    case NameContext::State::Finished:
      CARBON_FATAL("Finished state should only be used internally");
  }
}

auto DeclNameStack::MakeEmptyNameContext() -> NameContext {
  return NameContext{
      .initial_scope_index = context_->scope_stack().PeekIndex(),
      .parent_scope_id = context_->scope_stack().PeekNameScopeId()};
}

auto DeclNameStack::MakeUnqualifiedName(SemIR::LocId loc_id,
                                        SemIR::NameId name_id) -> NameContext {
  NameContext context = MakeEmptyNameContext();
  ApplyAndLookupName(context, loc_id, name_id);
  return context;
}

auto DeclNameStack::PushScopeAndStartName() -> void {
  decl_name_stack_.push_back(MakeEmptyNameContext());

  // Create a scope for any parameters introduced in this name.
  context_->scope_stack().Push();
}

auto DeclNameStack::FinishName(const NameComponent& name) -> NameContext {
  CARBON_CHECK(decl_name_stack_.back().state != NameContext::State::Finished,
               "Finished name twice");

  ApplyAndLookupName(decl_name_stack_.back(), name.name_loc_id, name.name_id);

  NameContext result = decl_name_stack_.back();
  decl_name_stack_.back().state = NameContext::State::Finished;
  return result;
}

auto DeclNameStack::FinishImplName() -> NameContext {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Empty,
               "Impl has a name");

  NameContext result = decl_name_stack_.back();
  decl_name_stack_.back().state = NameContext::State::Finished;
  return result;
}

auto DeclNameStack::PopScope() -> void {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Finished,
               "Missing call to FinishName before PopScope");
  context_->scope_stack().PopTo(decl_name_stack_.back().initial_scope_index);
  decl_name_stack_.pop_back();
}

auto DeclNameStack::Suspend() -> SuspendedName {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Finished,
               "Missing call to FinishName before Suspend");
  SuspendedName result = {.name_context = decl_name_stack_.pop_back_val(),
                          .scopes = {}};
  auto scope_index = result.name_context.initial_scope_index;
  auto& scope_stack = context_->scope_stack();
  while (scope_stack.PeekIndex() > scope_index) {
    result.scopes.push_back(scope_stack.Suspend());
  }
  CARBON_CHECK(scope_stack.PeekIndex() == scope_index,
               "Scope index {0} does not enclose the current scope {1}",
               scope_index, scope_stack.PeekIndex());
  return result;
}

auto DeclNameStack::Restore(SuspendedName sus) -> void {
  // The parent state must be the same when a name is restored.
  CARBON_CHECK(context_->scope_stack().PeekIndex() ==
                   sus.name_context.initial_scope_index,
               "Name restored at the wrong position in the name stack.");

  // clang-tidy warns that the `std::move` below has no effect. While that's
  // true, this `move` defends against `NameContext` growing more state later.
  // NOLINTNEXTLINE(performance-move-const-arg)
  decl_name_stack_.push_back(std::move(sus.name_context));
  for (auto& suspended_scope : llvm::reverse(sus.scopes)) {
    // Reattempt to resolve the definition of the specific. The generic might
    // have been defined after we suspended this scope.
    if (suspended_scope.entry.specific_id.has_value()) {
      ResolveSpecificDefinition(*context_, sus.name_context.loc_id,
                                suspended_scope.entry.specific_id);
    }

    context_->scope_stack().Restore(std::move(suspended_scope));
  }
}

auto DeclNameStack::AddName(NameContext name_context, SemIR::InstId target_id,
                            SemIR::AccessKind access_kind) -> void {
  switch (name_context.state) {
    case NameContext::State::Error:
      return;

    case NameContext::State::Unresolved:
      if (!name_context.parent_scope_id.has_value()) {
        AddNameToLookup(*context_, name_context.name_id, target_id,
                        name_context.initial_scope_index);
      } else {
        auto& name_scope =
            context_->name_scopes().Get(name_context.parent_scope_id);
        if (name_context.has_qualifiers) {
          auto inst = context_->insts().Get(name_scope.inst_id());
          if (!inst.Is<SemIR::Namespace>()) {
            // TODO: Point at the declaration for the scoped entity.
            CARBON_DIAGNOSTIC(
                QualifiedDeclOutsideScopeEntity, Error,
                "out-of-line declaration requires a declaration in "
                "scoped entity");
            context_->emitter().Emit(name_context.loc_id,
                                     QualifiedDeclOutsideScopeEntity);
          }
        }

        // Exports are only tracked when the declaration is at the file-level
        // scope. Otherwise, it's in some other entity, such as a class.
        if (access_kind == SemIR::AccessKind::Public &&
            name_context.initial_scope_index == ScopeIndex::Package) {
          context_->exports().push_back(target_id);
        }

        name_scope.AddRequired({.name_id = name_context.name_id,
                                .result = SemIR::ScopeLookupResult::MakeFound(
                                    target_id, access_kind)});
      }
      break;

    default:
      CARBON_FATAL("Should not be calling AddName");
      break;
  }
}

auto DeclNameStack::AddNameOrDiagnose(NameContext name_context,
                                      SemIR::InstId target_id,
                                      SemIR::AccessKind access_kind) -> void {
  if (name_context.state == DeclNameStack::NameContext::State::Poisoned) {
    DiagnosePoisonedName(*context_, name_context.name_id_for_new_inst(),
                         name_context.poisoning_loc_id, name_context.loc_id);
  } else if (auto id = name_context.prev_inst_id(); id.has_value()) {
    DiagnoseDuplicateName(*context_, name_context.name_id, name_context.loc_id,
                          id);
  } else {
    AddName(name_context, target_id, access_kind);
  }
}

auto DeclNameStack::LookupOrAddName(NameContext name_context,
                                    SemIR::InstId target_id,
                                    SemIR::AccessKind access_kind)
    -> SemIR::ScopeLookupResult {
  if (name_context.state == NameContext::State::Poisoned) {
    return SemIR::ScopeLookupResult::MakePoisoned(
        name_context.poisoning_loc_id);
  }
  if (auto id = name_context.prev_inst_id(); id.has_value()) {
    return SemIR::ScopeLookupResult::MakeFound(id, access_kind);
  }
  AddName(name_context, target_id, access_kind);
  return SemIR::ScopeLookupResult::MakeNotFound();
}

// Push a scope corresponding to a name qualifier. For example, for
// `fn Class(T:! type).F(n: i32)` we will push the scope for `Class(T:! type)`
// between the scope containing the declaration of `T` and the scope
// containing the declaration of `n`.
static auto PushNameQualifierScope(Context& context, SemIRLoc loc,
                                   SemIR::InstId scope_inst_id,
                                   SemIR::NameScopeId scope_id,
                                   SemIR::GenericId generic_id,
                                   bool has_error = false) -> void {
  // If the qualifier has no parameters, we don't need to keep around a
  // parameter scope.
  context.scope_stack().PopIfEmpty();

  auto self_specific_id = SemIR::SpecificId::None;
  if (generic_id.has_value()) {
    self_specific_id = context.generics().GetSelfSpecific(generic_id);
    // When declaring a member of a generic, resolve the self specific.
    ResolveSpecificDefinition(context, loc, self_specific_id);
  }

  // Close the generic stack scope and open a new one for whatever comes after
  // the qualifier. As this is a qualifier it must not be the initial
  // declaration of the entity, so we treat it as a redeclaration.
  FinishGenericRedecl(context, generic_id);
  // What follows the qualifier will be a declaration. The signature of an
  // entity is also a declaration even if it is followed by curly braces
  // providing the definition.
  StartGenericDecl(context);

  context.scope_stack().Push(scope_inst_id, scope_id, self_specific_id,
                             has_error);

  // An interface also introduces its 'Self' parameter into scope, despite it
  // not being redeclared as part of the qualifier.
  if (auto interface_decl =
          context.insts().TryGetAs<SemIR::InterfaceDecl>(scope_inst_id)) {
    auto& interface = context.interfaces().Get(interface_decl->interface_id);
    context.scope_stack().AddCompileTimeBinding();
    context.scope_stack().PushCompileTimeBinding(interface.self_param_id);
  }

  // Enter a parameter scope in case the qualified name itself has parameters.
  context.scope_stack().Push();
}

auto DeclNameStack::ApplyNameQualifier(const NameComponent& name) -> void {
  auto& name_context = decl_name_stack_.back();
  ApplyAndLookupName(name_context, name.name_loc_id, name.name_id);
  name_context.has_qualifiers = true;

  // Resolve the qualifier as a scope and enter the new scope.
  auto [scope_id, generic_id] = ResolveAsScope(name_context, name);
  if (scope_id.has_value()) {
    PushNameQualifierScope(*context_, name_context.loc_id,
                           name_context.resolved_inst_id, scope_id, generic_id,
                           context_->name_scopes().Get(scope_id).has_error());
    name_context.parent_scope_id = scope_id;
  } else {
    name_context.state = NameContext::State::Error;
  }
}

auto DeclNameStack::ApplyAndLookupName(NameContext& name_context,
                                       SemIR::LocId loc_id,
                                       SemIR::NameId name_id) -> void {
  // Update the final name component.
  name_context.loc_id = loc_id;
  name_context.name_id = name_id;

  // Don't perform any more lookups after we hit an error. We still track the
  // final name, though.
  if (name_context.state == NameContext::State::Error) {
    return;
  }

  // For identifier nodes, we need to perform a lookup on the identifier.
  auto lookup_result = LookupNameInDecl(*context_, name_context.loc_id, name_id,
                                        name_context.parent_scope_id,
                                        name_context.initial_scope_index);
  if (lookup_result.is_poisoned()) {
    name_context.poisoning_loc_id = lookup_result.poisoning_loc_id();
    name_context.state = NameContext::State::Poisoned;
  } else if (!lookup_result.is_found()) {
    // Invalid indicates an unresolved name. Store it and return.
    name_context.state = NameContext::State::Unresolved;
  } else {
    // Store the resolved instruction and continue for the target scope
    // update.
    name_context.resolved_inst_id = lookup_result.target_inst_id();
    name_context.state = NameContext::State::Resolved;
  }
}

// Checks and returns whether name_context, which is used as a name qualifier,
// was successfully resolved. Issues a suitable diagnostic if not.
static auto CheckQualifierIsResolved(
    Context& context, const DeclNameStack::NameContext& name_context) -> bool {
  switch (name_context.state) {
    case DeclNameStack::NameContext::State::Empty:
      CARBON_FATAL("No qualifier to resolve");

    case DeclNameStack::NameContext::State::Resolved:
      return true;

    case DeclNameStack::NameContext::State::Poisoned:
    case DeclNameStack::NameContext::State::Unresolved:
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier failed to resolve.
      DiagnoseNameNotFound(context, name_context.loc_id, name_context.name_id);
      return false;

    case DeclNameStack::NameContext::State::Finished:
      CARBON_FATAL("Added a qualifier after calling FinishName");

    case DeclNameStack::NameContext::State::Error:
      // Already in an error state, so return without examining.
      return false;
  }
}

// Diagnose that a qualified declaration name specifies an incomplete class as
// its scope.
static auto DiagnoseQualifiedDeclInIncompleteClassScope(Context& context,
                                                        SemIRLoc loc,
                                                        SemIR::ClassId class_id)
    -> void {
  CARBON_DIAGNOSTIC(QualifiedDeclInIncompleteClassScope, Error,
                    "cannot declare a member of incomplete class {0}",
                    SemIR::TypeId);
  auto builder =
      context.emitter().Build(loc, QualifiedDeclInIncompleteClassScope,
                              context.classes().Get(class_id).self_type_id);
  NoteIncompleteClass(context, class_id, builder);
  builder.Emit();
}

// Diagnose that a qualified declaration name specifies an undefined interface
// as its scope.
static auto DiagnoseQualifiedDeclInUndefinedInterfaceScope(
    Context& context, SemIRLoc loc, SemIR::InterfaceId interface_id,
    SemIR::InstId interface_inst_id) -> void {
  CARBON_DIAGNOSTIC(QualifiedDeclInUndefinedInterfaceScope, Error,
                    "cannot declare a member of undefined interface {0}",
                    InstIdAsType);
  auto builder = context.emitter().Build(
      loc, QualifiedDeclInUndefinedInterfaceScope, interface_inst_id);
  NoteUndefinedInterface(context, interface_id, builder);
  builder.Emit();
}

// Diagnose that a qualified declaration name specifies a different package as
// its scope.
static auto DiagnoseQualifiedDeclInImportedPackage(Context& context,
                                                   SemIRLoc use_loc,
                                                   SemIRLoc import_loc)
    -> void {
  CARBON_DIAGNOSTIC(QualifiedDeclOutsidePackage, Error,
                    "imported packages cannot be used for declarations");
  CARBON_DIAGNOSTIC(QualifiedDeclOutsidePackageSource, Note,
                    "package imported here");
  context.emitter()
      .Build(use_loc, QualifiedDeclOutsidePackage)
      .Note(import_loc, QualifiedDeclOutsidePackageSource)
      .Emit();
}

// Diagnose that a qualified declaration name specifies a non-scope entity as
// its scope.
static auto DiagnoseQualifiedDeclInNonScope(Context& context, SemIRLoc use_loc,
                                            SemIRLoc non_scope_entity_loc)
    -> void {
  CARBON_DIAGNOSTIC(QualifiedNameInNonScope, Error,
                    "name qualifiers are only allowed for entities that "
                    "provide a scope");
  CARBON_DIAGNOSTIC(QualifiedNameNonScopeEntity, Note,
                    "referenced non-scope entity declared here");
  context.emitter()
      .Build(use_loc, QualifiedNameInNonScope)
      .Note(non_scope_entity_loc, QualifiedNameNonScopeEntity)
      .Emit();
}

auto DeclNameStack::ResolveAsScope(const NameContext& name_context,
                                   const NameComponent& name) const
    -> std::pair<SemIR::NameScopeId, SemIR::GenericId> {
  constexpr std::pair<SemIR::NameScopeId, SemIR::GenericId> InvalidResult = {
      SemIR::NameScopeId::None, SemIR::GenericId::None};

  if (!CheckQualifierIsResolved(*context_, name_context)) {
    return InvalidResult;
  }

  if (name_context.state == NameContext::State::Poisoned) {
    return InvalidResult;
  }

  auto new_params = DeclParams(
      name.name_loc_id, name.first_param_node_id, name.last_param_node_id,
      name.implicit_param_patterns_id, name.param_patterns_id);

  // Find the scope corresponding to the resolved instruction.
  // TODO: When diagnosing qualifiers on names, print a diagnostic that talks
  // about qualifiers instead of redeclarations. Maybe also rename
  // CheckRedeclParamsMatch.
  CARBON_KIND_SWITCH(context_->insts().Get(name_context.resolved_inst_id)) {
    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      const auto& class_info = context_->classes().Get(class_decl.class_id);
      if (!CheckRedeclParamsMatch(*context_, new_params,
                                  DeclParams(class_info))) {
        return InvalidResult;
      }
      if (!class_info.is_defined()) {
        DiagnoseQualifiedDeclInIncompleteClassScope(
            *context_, name_context.loc_id, class_decl.class_id);
        return InvalidResult;
      }
      return {class_info.scope_id, class_info.generic_id};
    }
    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      const auto& interface_info =
          context_->interfaces().Get(interface_decl.interface_id);
      if (!CheckRedeclParamsMatch(*context_, new_params,
                                  DeclParams(interface_info))) {
        return InvalidResult;
      }
      if (!interface_info.is_defined()) {
        DiagnoseQualifiedDeclInUndefinedInterfaceScope(
            *context_, name_context.loc_id, interface_decl.interface_id,
            name_context.resolved_inst_id);
        return InvalidResult;
      }
      return {interface_info.scope_id, interface_info.generic_id};
    }
    case CARBON_KIND(SemIR::Namespace resolved_inst): {
      auto scope_id = resolved_inst.name_scope_id;
      auto& scope = context_->name_scopes().Get(scope_id);
      // This is specifically for qualified name handling.
      if (!CheckRedeclParamsMatch(
              *context_, new_params,
              DeclParams(name_context.resolved_inst_id, Parse::NodeId::None,
                         Parse::NodeId::None, SemIR::InstBlockId::None,
                         SemIR::InstBlockId::None))) {
        return InvalidResult;
      }
      if (scope.is_closed_import()) {
        DiagnoseQualifiedDeclInImportedPackage(*context_, name_context.loc_id,
                                               scope.inst_id());
        // Only error once per package. Recover by allowing this package name to
        // be used as a name qualifier.
        scope.set_is_closed_import(false);
      }
      return {scope_id, SemIR::GenericId::None};
    }
    default: {
      DiagnoseQualifiedDeclInNonScope(*context_, name_context.loc_id,
                                      name_context.resolved_inst_id);
      return InvalidResult;
    }
  }
}

}  // namespace Carbon::Check
