// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles processing of a type's introducer.
static auto ParserHandleTypeIntroducer(ParserContext& context,
                                       ParseNodeKind introducer_kind,
                                       ParserState after_params_state) -> void {
  auto state = context.PopState();

  context.AddLeafNode(introducer_kind, context.Consume());

  state.state = after_params_state;
  context.PushState(state);
  state.state = ParserState::DeclarationNameAndParamsAsOptional;
  context.PushState(state);
}

auto ParserHandleTypeIntroducerAsClass(ParserContext& context) -> void {
  ParserHandleTypeIntroducer(context, ParseNodeKind::ClassIntroducer,
                             ParserState::TypeAfterParamsAsClass);
}

auto ParserHandleTypeIntroducerAsInterface(ParserContext& context) -> void {
  ParserHandleTypeIntroducer(context, ParseNodeKind::InterfaceIntroducer,
                             ParserState::TypeAfterParamsAsInterface);
}

auto ParserHandleTypeIntroducerAsNamedConstraint(ParserContext& context)
    -> void {
  ParserHandleTypeIntroducer(context, ParseNodeKind::NamedConstraintIntroducer,
                             ParserState::TypeAfterParamsAsNamedConstraint);
}

// Handles processing after params, deciding whether it's a declaration or
// definition.
static auto ParserHandleTypeAfterParams(ParserContext& context,
                                        ParseNodeKind declaration_kind,
                                        ParseNodeKind definition_start_kind,
                                        ParserState definition_finish_state)
    -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.RecoverFromDeclarationError(state, declaration_kind,
                                        /*skip_past_likely_end=*/true);
    return;
  }

  if (auto semi = context.ConsumeIf(TokenKind::Semi)) {
    context.AddNode(declaration_kind, *semi, state.subtree_start,
                    state.has_error);
    return;
  }

  if (!context.PositionIs(TokenKind::OpenCurlyBrace)) {
    context.emitter().Emit(*context.position(),
                           ExpectedDeclarationSemiOrDefinition,
                           context.tokens().GetKind(state.token));
    context.RecoverFromDeclarationError(state, declaration_kind,
                                        /*skip_past_likely_end=*/true);
    return;
  }

  state.state = definition_finish_state;
  context.PushState(state);
  context.PushState(ParserState::DeclarationScopeLoop);
  context.AddNode(definition_start_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto ParserHandleTypeAfterParamsAsClass(ParserContext& context) -> void {
  ParserHandleTypeAfterParams(context, ParseNodeKind::ClassDeclaration,
                              ParseNodeKind::ClassDefinitionStart,
                              ParserState::TypeDefinitionFinishAsClass);
}

auto ParserHandleTypeAfterParamsAsInterface(ParserContext& context) -> void {
  ParserHandleTypeAfterParams(context, ParseNodeKind::InterfaceDeclaration,
                              ParseNodeKind::InterfaceDefinitionStart,
                              ParserState::TypeDefinitionFinishAsInterface);
}

auto ParserHandleTypeAfterParamsAsNamedConstraint(ParserContext& context)
    -> void {
  ParserHandleTypeAfterParams(
      context, ParseNodeKind::NamedConstraintDeclaration,
      ParseNodeKind::NamedConstraintDefinitionStart,
      ParserState::TypeDefinitionFinishAsNamedConstraint);
}

// Handles parsing after the declaration scope of a type.
static auto ParserHandleTypeDefinitionFinish(ParserContext& context,
                                             ParseNodeKind definition_kind)
    -> void {
  auto state = context.PopState();

  context.AddNode(definition_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto ParserHandleTypeDefinitionFinishAsClass(ParserContext& context) -> void {
  ParserHandleTypeDefinitionFinish(context, ParseNodeKind::ClassDefinition);
}

auto ParserHandleTypeDefinitionFinishAsInterface(ParserContext& context)
    -> void {
  ParserHandleTypeDefinitionFinish(context, ParseNodeKind::InterfaceDefinition);
}

auto ParserHandleTypeDefinitionFinishAsNamedConstraint(ParserContext& context)
    -> void {
  ParserHandleTypeDefinitionFinish(context,
                                   ParseNodeKind::NamedConstraintDefinition);
}

}  // namespace Carbon
