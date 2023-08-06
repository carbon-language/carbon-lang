// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_declaration_name_stack.h"

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsDeclarationNameStack::Push() -> void {
  declaration_name_stack_.push_back(
      {.state = Context::State::New,
       .target_scope_id = SemanticsNameScopeId::Invalid,
       .resolved_node_id = SemanticsNodeId::Invalid});
}

auto SemanticsDeclarationNameStack::Pop() -> Context {
  if (context_->parse_tree().node_kind(
          context_->node_stack().PeekParseNode()) ==
      ParseNodeKind::QualifiedDeclaration) {
    // Any parts from a QualifiedDeclaration will already have been processed
    // into the name.
    context_->node_stack()
        .PopAndDiscardSoloParseNode<ParseNodeKind::QualifiedDeclaration>();
  } else {
    // The name had no qualifiers, so we need to process the node now.
    auto [parse_node, name_id] =
        context_->node_stack().PopWithParseNode<ParseNodeKind::Name>();
    ApplyNameQualifier(parse_node, name_id);
  }

  return declaration_name_stack_.pop_back_val();
}

auto SemanticsDeclarationNameStack::AddNameToLookup(Context name_context,
                                                    SemanticsNodeId target_id)
    -> void {
  switch (name_context.state) {
    case Context::State::Error:
      // The name is invalid and a diagnostic has already been emitted.
      return;

    case Context::State::New:
      CARBON_FATAL() << "Name is missing, not expected to call AddNameToLookup "
                        "(but that may change based on error handling).";

    case Context::State::Resolved:
    case Context::State::ResolvedNonScope: {
      context_->DiagnoseDuplicateName(name_context.parse_node,
                                      name_context.resolved_node_id);
      return;
    }

    case Context::State::Unresolved:
      if (name_context.target_scope_id == SemanticsNameScopeId::Invalid) {
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

auto SemanticsDeclarationNameStack::ApplyExpressionQualifier(
    ParseTree::Node parse_node, SemanticsNodeId node_id) -> void {
  auto& name_context = declaration_name_stack_.back();
  if (CanResolveQualifier(name_context, parse_node)) {
    if (node_id == SemanticsNodeId::BuiltinError) {
      // The input node is an error, so error the context.
      name_context.state = Context::State::Error;
      return;
    }

    // For other nodes, we expect a regular resolved node, for example a
    // namespace or generic type. Store it and continue for the target scope
    // update.
    name_context.resolved_node_id = node_id;

    UpdateScopeIfNeeded(name_context);
  }
}

auto SemanticsDeclarationNameStack::ApplyNameQualifier(
    ParseTree::Node parse_node, SemanticsStringId name_id) -> void {
  auto& name_context = declaration_name_stack_.back();
  if (CanResolveQualifier(name_context, parse_node)) {
    // For identifier nodes, we need to perform a lookup on the identifier.
    // This means the input node_id is actually a string ID.
    auto resolved_node_id = context_->LookupName(
        name_context.parse_node, name_id, name_context.target_scope_id,
        /*print_diagnostics=*/false);
    if (resolved_node_id == SemanticsNodeId::BuiltinError) {
      // Invalid indicates an unresolved node. Store it and return.
      name_context.state = Context::State::Unresolved;
      name_context.unresolved_name_id = name_id;
      return;
    } else {
      // Store the resolved node and continue for the target scope update.
      name_context.resolved_node_id = resolved_node_id;
    }

    UpdateScopeIfNeeded(name_context);
  }
}

auto SemanticsDeclarationNameStack::UpdateScopeIfNeeded(Context& name_context)
    -> void {
  // This will only be reached for resolved nodes. We update the target
  // scope based on the resolved type.
  auto resolved_node =
      context_->semantics_ir().GetNode(name_context.resolved_node_id);
  switch (resolved_node.kind()) {
    case SemanticsNodeKind::Namespace:
      name_context.state = Context::State::Resolved;
      name_context.target_scope_id = resolved_node.GetAsNamespace();
      break;
    default:
      name_context.state = Context::State::ResolvedNonScope;
      break;
  }
}

auto SemanticsDeclarationNameStack::CanResolveQualifier(
    Context& name_context, ParseTree::Node parse_node) -> bool {
  switch (name_context.state) {
    case Context::State::Error:
      // Already in an error state, so return without examining.
      return false;

    case Context::State::Unresolved:
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier failed to resolve.
      name_context.state = Context::State::Error;
      context_->DiagnoseNameNotFound(name_context.parse_node,
                                     name_context.unresolved_name_id);
      return false;

    case Context::State::ResolvedNonScope: {
      // Because more qualifiers were found, we diagnose that the earlier
      // qualifier didn't resolve to a scoped entity.
      name_context.state = Context::State::Error;
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

    case Context::State::New:
    case Context::State::Resolved: {
      name_context.parse_node = parse_node;
      return true;
    }
  }
}

}  // namespace Carbon
