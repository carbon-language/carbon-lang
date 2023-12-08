// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles processing of a type declaration or definition after its introducer.
static auto HandleTypeAfterIntroducer(Context& context,
                                      State after_params_state) -> void {
  auto state = context.PopState();
  context.PushState(state, after_params_state);
  context.PushState(State::DeclNameAndParamsAsOptional, state.token);
}

auto HandleTypeAfterIntroducerAsClass(Context& context) -> void {
  HandleTypeAfterIntroducer(context, State::TypeAfterParamsAsClass);
}

auto HandleTypeAfterIntroducerAsInterface(Context& context) -> void {
  HandleTypeAfterIntroducer(context, State::TypeAfterParamsAsInterface);
}

auto HandleTypeAfterIntroducerAsNamedConstraint(Context& context) -> void {
  HandleTypeAfterIntroducer(context, State::TypeAfterParamsAsNamedConstraint);
}

// Handles processing after params, deciding whether it's a declaration or
// definition.
static auto HandleTypeAfterParams(Context& context, NodeKind decl_kind,
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
    context.EmitExpectedDeclSemiOrDefinition(
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

auto HandleTypeAfterParamsAsClass(Context& context) -> void {
  HandleTypeAfterParams(context, NodeKind::ClassDecl,
                        NodeKind::ClassDefinitionStart,
                        State::TypeDefinitionFinishAsClass);
}

auto HandleTypeAfterParamsAsInterface(Context& context) -> void {
  HandleTypeAfterParams(context, NodeKind::InterfaceDecl,
                        NodeKind::InterfaceDefinitionStart,
                        State::TypeDefinitionFinishAsInterface);
}

auto HandleTypeAfterParamsAsNamedConstraint(Context& context) -> void {
  HandleTypeAfterParams(context, NodeKind::NamedConstraintDecl,
                        NodeKind::NamedConstraintDefinitionStart,
                        State::TypeDefinitionFinishAsNamedConstraint);
}

// Handles parsing after the declaration scope of a type.
static auto HandleTypeDefinitionFinish(Context& context,
                                       NodeKind definition_kind) -> void {
  auto state = context.PopState();

  context.AddNode(definition_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleTypeDefinitionFinishAsClass(Context& context) -> void {
  HandleTypeDefinitionFinish(context, NodeKind::ClassDefinition);
}

auto HandleTypeDefinitionFinishAsInterface(Context& context) -> void {
  HandleTypeDefinitionFinish(context, NodeKind::InterfaceDefinition);
}

auto HandleTypeDefinitionFinishAsNamedConstraint(Context& context) -> void {
  HandleTypeDefinitionFinish(context, NodeKind::NamedConstraintDefinition);
}

}  // namespace Carbon::Parse
