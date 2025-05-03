// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {
auto HandleChoiceIntroducer(Context& context) -> void {
  auto state = context.PopState();

  context.PushState(state, StateKind::ChoiceDefinitionStart);
  context.PushState(StateKind::DeclNameAndParams, state.token);
}

auto HandleChoiceDefinitionStart(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedChoiceDefinition, Error,
                        "choice definition expected");
      context.emitter().Emit(*context.position(), ExpectedChoiceDefinition);
    }

    context.AddNode(NodeKind::ChoiceDefinitionStart, *context.position(),
                    /*has_error=*/true);

    context.AddNode(NodeKind::ChoiceDefinition, *context.position(),
                    /*has_error=*/true);

    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddNode(NodeKind::ChoiceDefinitionStart, context.Consume(),
                  state.has_error);

  state.has_error = false;
  state.kind = StateKind::ChoiceDefinitionFinish;
  context.PushState(state);

  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushState(StateKind::ChoiceAlternative);
  }
}

auto HandleChoiceAlternative(Context& context) -> void {
  auto state = context.PopState();

  context.PushState(StateKind::ChoiceAlternativeFinish);

  auto token = context.ConsumeIf(Lex::TokenKind::Identifier);
  if (!token) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedChoiceAlternativeName, Error,
                        "expected choice alternative name");
      context.emitter().Emit(*context.position(),
                             ExpectedChoiceAlternativeName);
    }

    context.SkipPastLikelyEnd(*context.position());

    context.ReturnErrorOnState();

    return;
  }

  if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.AddLeafNode(NodeKind::IdentifierNameBeforeParams, *token);
    context.PushState(StateKind::PatternListAsExplicit);
  } else {
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *token);
  }
}

auto HandleChoiceAlternativeFinish(Context& context) -> void {
  const auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
    if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
      context.PushState(StateKind::ChoiceAlternative);
    }
    return;
  }

  if (context.ConsumeListToken(
          NodeKind::ChoiceAlternativeListComma, Lex::TokenKind::CloseCurlyBrace,
          state.has_error) == Context::ListTokenKind::Comma) {
    context.PushState(StateKind::ChoiceAlternative);
  }
}

auto HandleChoiceDefinitionFinish(Context& context) -> void {
  const auto state = context.PopState();

  context.AddNode(NodeKind::ChoiceDefinition,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.has_error);
}
}  // namespace Carbon::Parse
