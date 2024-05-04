// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles processing after params, deciding whether it's a declaration or
// definition.
static auto HandleDeclOrDefinition(Context& context, NodeKind decl_kind,
                                   NodeKind definition_start_kind,
                                   State definition_finish_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.RecoverFromDeclError(state, decl_kind,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    context.AddNode(decl_kind, *semi, state.subtree_start, state.has_error);
    return;
  }

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    context.DiagnoseExpectedDeclSemiOrDefinition(
        context.tokens().GetKind(state.token));
    context.RecoverFromDeclError(state, decl_kind,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  context.PushState(state, definition_finish_state);
  context.PushState(State::DeclScopeLoop);
  context.AddNode(definition_start_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleDeclOrDefinitionAsClass(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::ClassDecl,
                         NodeKind::ClassDefinitionStart,
                         State::DeclDefinitionFinishAsClass);
}

auto HandleDeclOrDefinitionAsImpl(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::ImplDecl,
                         NodeKind::ImplDefinitionStart,
                         State::DeclDefinitionFinishAsImpl);
}

auto HandleDeclOrDefinitionAsInterface(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::InterfaceDecl,
                         NodeKind::InterfaceDefinitionStart,
                         State::DeclDefinitionFinishAsInterface);
}

auto HandleDeclOrDefinitionAsNamedConstraint(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::NamedConstraintDecl,
                         NodeKind::NamedConstraintDefinitionStart,
                         State::DeclDefinitionFinishAsNamedConstraint);
}

// Handles parsing after the declaration scope of a type.
static auto HandleDeclDefinitionFinish(Context& context,
                                       NodeKind definition_kind) -> void {
  auto state = context.PopState();

  context.AddNode(definition_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleDeclDefinitionFinishAsClass(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::ClassDefinition);
}

auto HandleDeclDefinitionFinishAsImpl(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::ImplDefinition);
}

auto HandleDeclDefinitionFinishAsInterface(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::InterfaceDefinition);
}

auto HandleDeclDefinitionFinishAsNamedConstraint(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::NamedConstraintDefinition);
}

}  // namespace Carbon::Parse
