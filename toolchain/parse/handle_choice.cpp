// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {
auto HandleChoiceIntroducer(Context& context) -> void {
  auto state = context.PopState();

  context.PushState(state, StateKind::ChoiceDefinitionStart);
  // Choice alternative parameters, parsed separately below, are runtime by
  // default, unlike the choice type parameters handled here.
  context.PushState(StateKind::DeclNameAndParams, state.token,
                    BindingContext::CompileTimeEntityParam);
}

auto HandleChoiceDefinitionStart(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedChoiceDefinition, Error,
                        "choice definition expected");
      context.emitter().Emit(context.OpeningBracketInsertionPoint(),
                             ExpectedChoiceDefinition);
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
    // A word that can't be a name, followed by a token that can follow one,
    // was most likely meant as the name.
    bool recover_as_raw =
        context.PositionKind().is_word() &&
        context.PositionKind(Lookahead::NextToken)
            .IsOneOf({Lex::TokenKind::Comma, Lex::TokenKind::CloseCurlyBrace,
                      Lex::TokenKind::OpenParen});
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(
          ExpectedChoiceAlternativeName, Error,
          "expected choice alternative name"
          "{0:; prefix reserved word with `r#` to form a valid identifier|}",
          Diagnostics::BoolAsSelect);
      context.emitter().Emit(*context.position(), ExpectedChoiceAlternativeName,
                             recover_as_raw);
    }

    if (!recover_as_raw) {
      context.SkipPastLikelyEnd(*context.position());
      context.ReturnErrorOnState();
      return;
    }
    token = context.tokens().AddPostLexingRecoveryTokenAsIdentifier(
        context.Consume());
  }

  if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.AddLeafNode(NodeKind::IdentifierNameMaybeBeforeSignature, *token);
    context.PushState(StateKind::PatternListAsExplicit);
  } else {
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeSignature, *token);
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
