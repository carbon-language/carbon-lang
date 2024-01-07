// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {
auto HandleChoiceIntroducer(Context& context) -> void {
  auto state = context.PopState();

  state.state = State::ChoiceDefinitionStart;
  context.PushState(state);
  context.PushState(State::DeclNameAndParamsAsOptional, state.token);
}

auto HandleChoiceDefinitionStart(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedChoiceDefinition, Error,
                        "Expected Choice definition.");
      context.emitter().Emit(*context.position(), ExpectedChoiceDefinition);
    }
    // Only need to skip if we've not already found a new line.
    const bool skip_past_likely_end =
        context.tokens().GetLine(*context.position()) ==
        context.tokens().GetLine(state.token);
    context.RecoverFromDeclError(state, NodeKind::ChoiceIntroducer,
                                 skip_past_likely_end);
    return;
  }

  context.AddNode(NodeKind::ChoiceDefinitionStart, context.Consume(),
                  state.subtree_start, state.has_error);

  state.has_error = false;
  state.state = State::ChoiceDefinitionFinish;
  context.PushState(state);

  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushState(State::ChoiceAlternative);
  }
}

auto HandleChoiceAlternative(Context& context) -> void {
  auto state = context.PopState();
  if (!context.PositionIs(Lex::TokenKind::Identifier)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedChoiceAlternativeName, Error,
                        "Expected Choice alternative name.");
      context.emitter().Emit(*context.position(),
                             ExpectedChoiceAlternativeName);
    }

    // TODO: recover?
    CARBON_CHECK(false) << "Not implemented";
    return;
  }

  context.PushState(State::ChoiceAlternativeFinish);
  context.PushState(State::DeclNameAndParamsAsOptional, state.token);
}

auto HandleChoiceAlternativeFinish(Context& context) -> void {
  const auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(
          NodeKind::ChoiceAlternativeListComma, Lex::TokenKind::CloseCurlyBrace,
          state.has_error) == Context::ListTokenKind::Comma) {
    context.PushState(State::ChoiceAlternative);
  }
}

auto HandleChoiceDefinitionFinish(Context& context) -> void {
  const auto state = context.PopState();
  context.AddNode(NodeKind::ChoiceDefinition,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.subtree_start, state.has_error);
}
}  // namespace Carbon::Parse
