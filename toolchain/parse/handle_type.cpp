// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles processing of a type's introducer.
static auto HandleTypeIntroducer(Context& context, LampKind introducer_kind,
                                 State after_params_state) -> void {
  auto state = context.PopState();

  context.AddLeafNode(introducer_kind, context.Consume());

  state.state = after_params_state;
  context.PushState(state);
  context.PushState(State::DeclarationNameAndParamsAsOptional, state.token);
}

auto HandleTypeIntroducerAsClass(Context& context) -> void {
  HandleTypeIntroducer(context, LampKind::ClassIntroducer,
                       State::TypeAfterParamsAsClass);
}

auto HandleTypeIntroducerAsInterface(Context& context) -> void {
  HandleTypeIntroducer(context, LampKind::InterfaceIntroducer,
                       State::TypeAfterParamsAsInterface);
}

auto HandleTypeIntroducerAsNamedConstraint(Context& context) -> void {
  HandleTypeIntroducer(context, LampKind::NamedConstraintIntroducer,
                       State::TypeAfterParamsAsNamedConstraint);
}

// Handles processing after params, deciding whether it's a declaration or
// definition.
static auto HandleTypeAfterParams(Context& context, LampKind declaration_kind,
                                  LampKind definition_start_kind,
                                  State definition_finish_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.RecoverFromDeclarationError(state, declaration_kind,
                                        /*skip_past_likely_end=*/true);
    return;
  }

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    context.AddNode(declaration_kind, *semi, state.subtree_start,
                    state.has_error);
    return;
  }

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    context.EmitExpectedDeclarationSemiOrDefinition(
        context.tokens().GetKind(state.token));
    context.RecoverFromDeclarationError(state, declaration_kind,
                                        /*skip_past_likely_end=*/true);
    return;
  }

  state.state = definition_finish_state;
  context.PushState(state);
  context.PushState(State::DeclarationScopeLoop);
  context.AddNode(definition_start_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleTypeAfterParamsAsClass(Context& context) -> void {
  HandleTypeAfterParams(context, LampKind::ClassDeclaration,
                        LampKind::ClassDefinitionStart,
                        State::TypeDefinitionFinishAsClass);
}

auto HandleTypeAfterParamsAsInterface(Context& context) -> void {
  HandleTypeAfterParams(context, LampKind::InterfaceDeclaration,
                        LampKind::InterfaceDefinitionStart,
                        State::TypeDefinitionFinishAsInterface);
}

auto HandleTypeAfterParamsAsNamedConstraint(Context& context) -> void {
  HandleTypeAfterParams(context, LampKind::NamedConstraintDeclaration,
                        LampKind::NamedConstraintDefinitionStart,
                        State::TypeDefinitionFinishAsNamedConstraint);
}

// Handles parsing after the declaration scope of a type.
static auto HandleTypeDefinitionFinish(Context& context,
                                       LampKind definition_kind) -> void {
  auto state = context.PopState();

  context.AddNode(definition_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleTypeDefinitionFinishAsClass(Context& context) -> void {
  HandleTypeDefinitionFinish(context, LampKind::ClassDefinition);
}

auto HandleTypeDefinitionFinishAsInterface(Context& context) -> void {
  HandleTypeDefinitionFinish(context, LampKind::InterfaceDefinition);
}

auto HandleTypeDefinitionFinishAsNamedConstraint(Context& context) -> void {
  HandleTypeDefinitionFinish(context, LampKind::NamedConstraintDefinition);
}

}  // namespace Carbon::Parse
