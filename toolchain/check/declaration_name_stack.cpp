// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/declaration_name_stack.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto DeclarationNameStack::Push() -> void {
  declaration_name_stack_.push_back(
      {.state = NameContext::State::New,
       .target_scope_id = SemIR::NameScopeId::Invalid,
       .resolved_node_id = SemIR::NodeId::Invalid});
}

auto DeclarationNameStack::Pop() -> NameContext {
  if (context_->parse_tree().node_kind(
          context_->node_stack().PeekParseNode()) ==
      Parse::NodeKind::QualifiedDeclaration) {
    // Any parts from a QualifiedDeclaration will already have been processed
    // into the name.
    context_->node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::QualifiedDeclaration>();
  } else {
    // The name had no qualifiers, so we need to process the node now.
    auto [parse_node, name_id] =
        context_->node_stack().PopWithParseNode<Parse::NodeKind::Name>();
    ApplyNameQualifier(parse_node, name_id);
  }

  return declaration_name_stack_.pop_back_val();
}

auto DeclarationNameStack::AddNameToLookup(NameContext name_context,
                                           SemIR::NodeId target_id) -> void {
  switch (name_context.state) {
    case NameContext::State::Error:
      // The name is invalid and a diagnostic has already been emitted.
      return;

    case NameContext::State::New:
      CARBON_FATAL() << "Name is missing, not expected to call AddNameToLookup "
                        "(but that may change based on error handling).";

    case NameContext::State::Resolved:
    case NameContext::State::ResolvedNonScope: {
      context_->DiagnoseDuplicateName(name_context.parse_node,
                                      name_context.resolved_node_id);
      return;
    }

    case NameContext::State::Unresolved:
      if (name_context.target_scope_id == SemIR::NameScopeId::Invalid) {
        context_->AddNameToLookup(name_context.parse_node,
                                  name_context.unresolved_name_id, target_id);
      } else {
        bool success = context_->semantics_ir().AddNameScopeEntry(
            name_context.target_scope_id, name_context.unresolved_name_id,
            target_id);
        CARBON_CHECK(success)
            << "Duplicate names should have been resolved previously: "
            << name_context.unresolved_name_id << " in "
            << name_context.target_scope_id;
      }
      return;
  }
}

auto DeclarationNameStack::ApplyExpressionQualifier(Parse::Node parse_node,
                                                    SemIR::NodeId node_id)
    -> void {
  auto& name_context = declaration_name_stack_.back();
  if (CanResolveQualifier(name_context, parse_node)) {
    if (node_id == SemIR::NodeId::BuiltinError) {
      // The input node is an error, so error the context.
      name_context.state = NameContext::State::Error;
      return;
    }

    // For other nodes, we expect a regular resolved node, for example a
    // namespace or generic type. Store it and continue for the target scope
    // update.
    name_context.resolved_node_id = node_id;

    UpdateScopeIfNeeded(name_context);
  }
}

auto DeclarationNameStack::ApplyNameQualifier(Parse::Node parse_node,
                                              SemIR::StringId name_id) -> void {
  auto& name_context = declaration_name_stack_.back();
  if (CanResolveQualifier(name_context, parse_node)) {
    // For identifier nodes, we need to perform a lookup on the identifier.
    // This means the input node_id is actually a string ID.
    auto resolved_node_id = context_->LookupName(
        name_context.parse_node, name_id, name_context.target_scope_id,
        /*print_diagnostics=*/false);
    if (resolved_node_id == SemIR::NodeId::BuiltinError) {
      // Invalid indicates an unresolved node. Store it and return.
      name_context.state = NameContext::State::Unresolved;
      name_context.unresolved_name_id = name_id;
      return;
    } else {
      // Store the resolved node and continue for the target scope update.
      name_context.resolved_node_id = resolved_node_id;
    }

    UpdateScopeIfNeeded(name_context);
  }
}

auto DeclarationNameStack::UpdateScopeIfNeeded(NameContext& name_context)
    -> void {
  // This will only be reached for resolved nodes. We update the target
  // scope based on the resolved type.
  auto resolved_node =
      context_->semantics_ir().GetNode(name_context.resolved_node_id);
  switch (resolved_node.kind()) {
    case SemIR::NodeKind::Namespace:
      name_context.state = NameContext::State::Resolved;
      name_context.target_scope_id = resolved_node.GetAsNamespace();
      break;
    default:
      name_context.state = NameContext::State::ResolvedNonScope;
      break;
  }
}

auto DeclarationNameStack::CanResolveQualifier(NameContext& name_context,
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
      name_context.state = NameContext::State::Error;
      CARBON_DIAGNOSTIC(QualifiedDeclarationInNonScope, Error,
                        "Declaration qualifiers are only allowed for entities "
                        "that provide a scope.");
      CARBON_DIAGNOSTIC(QualifiedDeclarationNonScopeEntity, Note,
                        "Non-scope entity referenced here.");
      context_->emitter()
          .Build(parse_node, QualifiedDeclarationInNonScope)
          .Note(name_context.parse_node, QualifiedDeclarationNonScopeEntity)
          .Emit();
      return false;
    }

    case NameContext::State::New:
    case NameContext::State::Resolved: {
      name_context.parse_node = parse_node;
      return true;
    }
  }
}

}  // namespace Carbon::Check
