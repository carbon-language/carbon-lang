// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/decl_name_stack.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto DeclNameStack::MakeEmptyNameContext() -> NameContext {
  return NameContext{.enclosing_scope = context_->current_scope_index(),
                     .target_scope_id = context_->current_scope_id()};
}

auto DeclNameStack::MakeUnqualifiedName(Parse::Node parse_node,
                                        SemIR::NameId name_id) -> NameContext {
  NameContext context = MakeEmptyNameContext();
  ApplyNameQualifierTo(context, parse_node, name_id);
  return context;
}

auto DeclNameStack::PushScopeAndStartName() -> void {
  decl_name_stack_.push_back(MakeEmptyNameContext());
}

auto DeclNameStack::FinishName() -> NameContext {
  CARBON_CHECK(decl_name_stack_.back().state != NameContext::State::Finished)
      << "Finished name twice";
  if (context_->parse_tree().node_kind(
          context_->node_stack().PeekParseNode()) ==
      Parse::NodeKind::QualifiedDecl) {
    // Any parts from a QualifiedDecl will already have been processed
    // into the name.
    context_->node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::QualifiedDecl>();
  } else {
    // The name had no qualifiers, so we need to process the node now.
    auto [parse_node, name_id] =
        context_->node_stack().PopWithParseNode<Parse::NodeKind::Name>();
    ApplyNameQualifier(parse_node, name_id);
  }

  NameContext result = decl_name_stack_.back();
  decl_name_stack_.back().state = NameContext::State::Finished;
  return result;
}

auto DeclNameStack::PopScope() -> void {
  CARBON_CHECK(decl_name_stack_.back().state == NameContext::State::Finished)
      << "Missing call to FinishName before PopScope";
  context_->PopToScope(decl_name_stack_.back().enclosing_scope);
  decl_name_stack_.pop_back();
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
      if (name_context.target_scope_id == SemIR::NameScopeId::Invalid) {
        context_->AddNameToLookup(name_context.parse_node,
                                  name_context.unresolved_name_id, target_id);
      } else {
        // TODO: Reject unless the scope is a namespace scope or the name is
        // unqualified.
        bool success = context_->name_scopes().AddEntry(
            name_context.target_scope_id, name_context.unresolved_name_id,
            target_id);
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
    context_->DiagnoseDuplicateName(name_context.parse_node, existing_inst_id);
  }
}

auto DeclNameStack::ApplyNameQualifier(Parse::Node parse_node,
                                       SemIR::NameId name_id) -> void {
  ApplyNameQualifierTo(decl_name_stack_.back(), parse_node, name_id);
}

auto DeclNameStack::ApplyNameQualifierTo(NameContext& name_context,
                                         Parse::Node parse_node,
                                         SemIR::NameId name_id) -> void {
  if (CanResolveQualifier(name_context, parse_node)) {
    // For identifier nodes, we need to perform a lookup on the identifier.
    auto resolved_inst_id = context_->LookupNameInDecl(
        name_context.parse_node, name_id, name_context.target_scope_id);
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

    UpdateScopeIfNeeded(name_context);
  }
}

auto DeclNameStack::UpdateScopeIfNeeded(NameContext& name_context) -> void {
  // This will only be reached for resolved instructions. We update the target
  // scope based on the resolved type.
  auto resolved_inst = context_->insts().Get(name_context.resolved_inst_id);
  switch (resolved_inst.kind()) {
    case SemIR::ClassDecl::Kind: {
      const auto& class_info = context_->classes().Get(
          resolved_inst.As<SemIR::ClassDecl>().class_id);
      if (class_info.is_defined()) {
        name_context.state = NameContext::State::Resolved;
        name_context.target_scope_id = class_info.scope_id;
        context_->PushScope(name_context.resolved_inst_id, class_info.scope_id);
      } else {
        name_context.state = NameContext::State::ResolvedNonScope;
      }
      break;
    }
    case SemIR::Namespace::Kind: {
      auto scope_id = resolved_inst.As<SemIR::Namespace>().name_scope_id;
      name_context.state = NameContext::State::Resolved;
      name_context.target_scope_id = scope_id;
      context_->PushScope(name_context.resolved_inst_id, scope_id);
      break;
    }
    default:
      name_context.state = NameContext::State::ResolvedNonScope;
      break;
  }
}

auto DeclNameStack::CanResolveQualifier(NameContext& name_context,
                                        Parse::Node parse_node) -> bool {
  switch (name_context.state) {
    case NameContext::State::Error:
      // Already in an error state, so return without examining.
      return false;

    case NameContext::State::Unresolved:
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier failed to resolve.
      name_context.state = NameContext::State::Error;
      context_->DiagnoseNameNotFound(name_context.parse_node,
                                     name_context.unresolved_name_id);
      return false;

    case NameContext::State::ResolvedNonScope: {
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier didn't resolve to a scoped entity.
      if (auto class_decl = context_->insts()
                                .Get(name_context.resolved_inst_id)
                                .TryAs<SemIR::ClassDecl>()) {
        CARBON_DIAGNOSTIC(QualifiedDeclInIncompleteClassScope, Error,
                          "Cannot declare a member of incomplete class `{0}`.",
                          std::string);
        auto builder = context_->emitter().Build(
            name_context.parse_node, QualifiedDeclInIncompleteClassScope,
            context_->sem_ir().StringifyType(
                context_->classes().Get(class_decl->class_id).self_type_id,
                true));
        context_->NoteIncompleteClass(class_decl->class_id, builder);
        builder.Emit();
      } else {
        CARBON_DIAGNOSTIC(
            QualifiedDeclInNonScope, Error,
            "Declaration qualifiers are only allowed for entities "
            "that provide a scope.");
        CARBON_DIAGNOSTIC(QualifiedDeclNonScopeEntity, Note,
                          "Non-scope entity referenced here.");
        context_->emitter()
            .Build(parse_node, QualifiedDeclInNonScope)
            .Note(name_context.parse_node, QualifiedDeclNonScopeEntity)
            .Emit();
      }
      name_context.state = NameContext::State::Error;
      return false;
    }

    case NameContext::State::Empty:
    case NameContext::State::Resolved: {
      name_context.parse_node = parse_node;
      return true;
    }

    case NameContext::State::Finished:
      CARBON_FATAL() << "Added a qualifier after calling FinishName";
  }
}

}  // namespace Carbon::Check
