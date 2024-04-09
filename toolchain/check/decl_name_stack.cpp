// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/decl_name_stack.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto DeclNameStack::MakeEmptyNameContext() -> NameContext {
  return NameContext{
      .enclosing_scope = context_->scope_stack().PeekIndex(),
      .target_scope_id = context_->scope_stack().PeekNameScopeId()};
}

auto DeclNameStack::MakeUnqualifiedName(SemIR::LocId loc_id,
                                        SemIR::NameId name_id) -> NameContext {
  NameContext context = MakeEmptyNameContext();
  ApplyNameQualifierTo(context, loc_id, name_id, /*is_unqualified=*/true);
  return context;
}

auto DeclNameStack::PushScopeAndStartName() -> void {
  decl_name_stack_.push_back(MakeEmptyNameContext());

  // Create a scope for any parameters introduced in this name.
  context_->scope_stack().Push();
}

auto DeclNameStack::FinishName() -> NameContext {
  CARBON_CHECK(decl_name_stack_.back().state != NameContext::State::Finished)
      << "Finished name twice";
  if (context_->node_stack()
          .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::QualifiedName>()) {
    // Any parts from a QualifiedName will already have been processed
    // into the name.
  } else {
    // The name had no qualifiers, so we need to process the node now.
    auto [loc_id, name_id] = context_->node_stack().PopNameWithNodeId();
    ApplyNameQualifier(loc_id, name_id);
  }

  NameContext result = decl_name_stack_.back();
  decl_name_stack_.back().state = NameContext::State::Finished;
  return result;
}

auto DeclNameStack::FinishImplName() -> NameContext {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Empty)
      << "Impl has a name";

  NameContext result = decl_name_stack_.back();
  decl_name_stack_.back().state = NameContext::State::Finished;
  return result;
}

auto DeclNameStack::PopScope() -> void {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Finished)
      << "Missing call to FinishName before PopScope";
  context_->scope_stack().PopTo(decl_name_stack_.back().enclosing_scope);
  decl_name_stack_.pop_back();
}

auto DeclNameStack::Suspend() -> SuspendedName {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Finished)
      << "Missing call to FinishName before Suspend";
  SuspendedName result = {decl_name_stack_.pop_back_val(), {}};
  auto enclosing_index = result.name_context.enclosing_scope;
  auto& scope_stack = context_->scope_stack();
  while (scope_stack.PeekIndex() > enclosing_index) {
    result.scopes.push_back(scope_stack.Suspend());
  }
  CARBON_CHECK(scope_stack.PeekIndex() == enclosing_index)
      << "Scope index " << enclosing_index
      << " does not enclose the current scope " << scope_stack.PeekIndex();
  return result;
}

auto DeclNameStack::Restore(SuspendedName sus) -> void {
  // The enclosing state must be the same when a name is restored.
  CARBON_CHECK(context_->scope_stack().PeekIndex() ==
               sus.name_context.enclosing_scope)
      << "Name restored at the wrong position in the name stack.";

  // clang-tidy warns that the `std::move` below has no effect. While that's
  // true, this `move` defends against `NameContext` growing more state later.
  // NOLINTNEXTLINE(performance-move-const-arg)
  decl_name_stack_.push_back(std::move(sus.name_context));
  for (auto& suspended_scope : llvm::reverse(sus.scopes)) {
    context_->scope_stack().Restore(std::move(suspended_scope));
  }
}

auto DeclNameStack::LookupOrAddName(NameContext name_context,
                                    SemIR::InstId target_id) -> SemIR::InstId {
  switch (name_context.state) {
    case NameContext::State::Error:
      // The name is invalid and a diagnostic has already been emitted.
      return SemIR::InstId::Invalid;

    case NameContext::State::Empty:
      CARBON_FATAL() << "Name is missing, not expected to call AddNameToLookup "
                        "(but that may change based on error handling).";

    case NameContext::State::Resolved:
    case NameContext::State::ResolvedNonScope:
      return name_context.resolved_inst_id;

    case NameContext::State::Unresolved:
      if (!name_context.target_scope_id.is_valid()) {
        context_->AddNameToLookup(name_context.unresolved_name_id, target_id);
      } else {
        auto& name_scope =
            context_->name_scopes().Get(name_context.target_scope_id);
        if (name_context.has_qualifiers) {
          auto inst = context_->insts().Get(name_scope.inst_id);
          if (!inst.Is<SemIR::Namespace>()) {
            // TODO: Point at the declaration for the scoped entity.
            CARBON_DIAGNOSTIC(
                QualifiedDeclOutsideScopeEntity, Error,
                "Out-of-line declaration requires a declaration in "
                "scoped entity.");
            context_->emitter().Emit(name_context.loc_id,
                                     QualifiedDeclOutsideScopeEntity);
          }
        }

        // Exports are only tracked when the declaration is at the file-level
        // scope. Otherwise, it's in some other entity, such as a class.
        if (name_context.enclosing_scope == ScopeIndex::Package) {
          context_->AddExport(target_id);
        }

        auto [_, success] = name_scope.names.insert(
            {name_context.unresolved_name_id, target_id});
        CARBON_CHECK(success)
            << "Duplicate names should have been resolved previously: "
            << name_context.unresolved_name_id << " in "
            << name_context.target_scope_id;
      }
      return SemIR::InstId::Invalid;

    case NameContext::State::Finished:
      CARBON_FATAL() << "Finished state should only be used internally";
  }
}

auto DeclNameStack::AddNameToLookup(NameContext name_context,
                                    SemIR::InstId target_id) -> void {
  auto existing_inst_id = LookupOrAddName(name_context, target_id);
  if (existing_inst_id.is_valid()) {
    context_->DiagnoseDuplicateName(target_id, existing_inst_id);
  }
}

auto DeclNameStack::ApplyNameQualifier(SemIR::LocId loc_id,
                                       SemIR::NameId name_id) -> void {
  ApplyNameQualifierTo(decl_name_stack_.back(), loc_id, name_id,
                       /*is_unqualified=*/false);
}

auto DeclNameStack::ApplyNameQualifierTo(NameContext& name_context,
                                         SemIR::LocId loc_id,
                                         SemIR::NameId name_id,
                                         bool is_unqualified) -> void {
  if (TryResolveQualifier(name_context, loc_id)) {
    // For identifier nodes, we need to perform a lookup on the identifier.
    auto resolved_inst_id = context_->LookupNameInDecl(
        name_context.loc_id, name_id, name_context.target_scope_id,
        /*mark_imports_used=*/false);
    if (!resolved_inst_id.is_valid()) {
      // Invalid indicates an unresolved name. Store it and return.
      name_context.state = NameContext::State::Unresolved;
      name_context.unresolved_name_id = name_id;
      return;
    } else {
      // Store the resolved instruction and continue for the target scope
      // update.
      name_context.resolved_inst_id = resolved_inst_id;
    }

    UpdateScopeIfNeeded(name_context, is_unqualified);
  }
}

// Push a scope corresponding to a name qualifier. For example, for
//
//   fn Class(T:! type).F(n: i32)
//
// we will push the scope for `Class(T:! type)` between the scope containing the
// declaration of `T` and the scope containing the declaration of `n`.
static auto PushNameQualifierScope(Context& context,
                                   SemIR::InstId scope_inst_id,
                                   SemIR::NameScopeId scope_id,
                                   bool has_error = false) -> void {
  // If the qualifier has no parameters, we don't need to keep around a
  // parameter scope.
  context.scope_stack().PopIfEmpty();

  context.scope_stack().Push(scope_inst_id, scope_id, has_error);

  // Enter a parameter scope in case the qualified name itself has parameters.
  context.scope_stack().Push();
}

auto DeclNameStack::UpdateScopeIfNeeded(NameContext& name_context,
                                        bool is_unqualified) -> void {
  // This will only be reached for resolved instructions. We update the target
  // scope based on the resolved type.
  CARBON_KIND_SWITCH(context_->insts().Get(name_context.resolved_inst_id)) {
    case CARBON_KIND(SemIR::ClassDecl resolved_inst): {
      const auto& class_info = context_->classes().Get(resolved_inst.class_id);
      if (class_info.is_defined()) {
        name_context.state = NameContext::State::Resolved;
        name_context.target_scope_id = class_info.scope_id;
        if (!is_unqualified) {
          PushNameQualifierScope(*context_, name_context.resolved_inst_id,
                                 class_info.scope_id);
        }
      } else {
        name_context.state = NameContext::State::ResolvedNonScope;
      }
      break;
    }
    case CARBON_KIND(SemIR::InterfaceDecl resolved_inst): {
      const auto& interface_info =
          context_->interfaces().Get(resolved_inst.interface_id);
      if (interface_info.is_defined()) {
        name_context.state = NameContext::State::Resolved;
        name_context.target_scope_id = interface_info.scope_id;
        if (!is_unqualified) {
          PushNameQualifierScope(*context_, name_context.resolved_inst_id,
                                 interface_info.scope_id);
        }
      } else {
        name_context.state = NameContext::State::ResolvedNonScope;
      }
      break;
    }
    case CARBON_KIND(SemIR::Namespace resolved_inst): {
      auto scope_id = resolved_inst.name_scope_id;
      name_context.state = NameContext::State::Resolved;
      name_context.target_scope_id = scope_id;
      auto& scope = context_->name_scopes().Get(scope_id);
      if (scope.is_closed_import) {
        CARBON_DIAGNOSTIC(QualifiedDeclOutsidePackage, Error,
                          "Imported packages cannot be used for declarations.");
        CARBON_DIAGNOSTIC(QualifiedDeclOutsidePackageSource, Note,
                          "Package imported here.");
        context_->emitter()
            .Build(name_context.loc_id, QualifiedDeclOutsidePackage)
            .Note(scope.inst_id, QualifiedDeclOutsidePackageSource)
            .Emit();
        // Only error once per package.
        scope.is_closed_import = false;
      }
      if (!is_unqualified) {
        PushNameQualifierScope(*context_, name_context.resolved_inst_id,
                               scope_id,
                               context_->name_scopes().Get(scope_id).has_error);
      }
      break;
    }
    default:
      name_context.state = NameContext::State::ResolvedNonScope;
      break;
  }
}

auto DeclNameStack::TryResolveQualifier(NameContext& name_context,
                                        SemIR::LocId loc_id) -> bool {
  // Update has_qualifiers based on the state before any possible changes. If
  // this is the first qualifier, it may just be the name.
  name_context.has_qualifiers = name_context.state != NameContext::State::Empty;

  switch (name_context.state) {
    case NameContext::State::Error:
      // Already in an error state, so return without examining.
      return false;

    case NameContext::State::Unresolved:
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier failed to resolve.
      name_context.state = NameContext::State::Error;
      context_->DiagnoseNameNotFound(name_context.loc_id,
                                     name_context.unresolved_name_id);
      return false;

    case NameContext::State::ResolvedNonScope: {
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier didn't resolve to a scoped entity.
      if (auto class_decl = context_->insts().TryGetAs<SemIR::ClassDecl>(
              name_context.resolved_inst_id)) {
        CARBON_DIAGNOSTIC(QualifiedDeclInIncompleteClassScope, Error,
                          "Cannot declare a member of incomplete class `{0}`.",
                          SemIR::TypeId);
        auto builder = context_->emitter().Build(
            name_context.loc_id, QualifiedDeclInIncompleteClassScope,
            context_->classes().Get(class_decl->class_id).self_type_id);
        context_->NoteIncompleteClass(class_decl->class_id, builder);
        builder.Emit();
      } else if (auto interface_decl =
                     context_->insts().TryGetAs<SemIR::InterfaceDecl>(
                         name_context.resolved_inst_id)) {
        CARBON_DIAGNOSTIC(
            QualifiedDeclInUndefinedInterfaceScope, Error,
            "Cannot declare a member of undefined interface `{0}`.",
            std::string);
        auto builder = context_->emitter().Build(
            name_context.loc_id, QualifiedDeclInUndefinedInterfaceScope,
            context_->sem_ir().StringifyTypeExpr(
                context_->sem_ir()
                    .constant_values()
                    .Get(name_context.resolved_inst_id)
                    .inst_id()));
        context_->NoteUndefinedInterface(interface_decl->interface_id, builder);
        builder.Emit();
      } else {
        CARBON_DIAGNOSTIC(QualifiedNameInNonScope, Error,
                          "Name qualifiers are only allowed for entities that "
                          "provide a scope.");
        CARBON_DIAGNOSTIC(QualifiedNameNonScopeEntity, Note,
                          "Non-scope entity referenced here.");
        context_->emitter()
            .Build(loc_id, QualifiedNameInNonScope)
            .Note(name_context.loc_id, QualifiedNameNonScopeEntity)
            .Emit();
      }
      name_context.state = NameContext::State::Error;
      return false;
    }

    case NameContext::State::Empty:
    case NameContext::State::Resolved: {
      name_context.loc_id = loc_id;
      return true;
    }

    case NameContext::State::Finished:
      CARBON_FATAL() << "Added a qualifier after calling FinishName";
  }
}

}  // namespace Carbon::Check
