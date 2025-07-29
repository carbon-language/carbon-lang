// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleLet(Context& context) -> void {
  auto state = context.PopState();

  // These will start at the `let`.
  context.PushState(state, StateKind::LetFinishAsLet);
  context.PushState(state, StateKind::LetAfterPatternAsLet);

  // This will start at the pattern.
  context.PushState(StateKind::Pattern);
}

auto HandleAssociatedConstantDecl(Context& context) -> void {
  auto state = context.PopState();

  auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier);
  if (!identifier) {
    CARBON_DIAGNOSTIC(ExpectedAssociatedConstantIdentifier, Error,
                      "expected identifier in associated constant declaration");
    context.emitter().Emit(*context.position(),
                           ExpectedAssociatedConstantIdentifier);
  }
  auto colon = context.ConsumeIf(Lex::TokenKind::ColonExclaim);
  if (identifier && !colon) {
    CARBON_DIAGNOSTIC(ExpectedAssociatedConstantColonExclaim, Error,
                      "expected `:!` in associated constant declaration");
    context.emitter().Emit(*context.position(),
                           ExpectedAssociatedConstantColonExclaim);
  }
  if (!identifier || !colon) {
    context.AddNode(NodeKind::AssociatedConstantDecl,
                    context.SkipPastLikelyEnd(*(context.position() - 1)),
                    /*has_error=*/true);
    state.has_error = true;
    return;
  }
  context.PushState(state, StateKind::LetFinishAsAssociatedConstant);
  context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
  state.token = *colon;
  context.PushState(state, StateKind::LetAfterPatternAsAssociatedConstant);
  context.PushState(StateKind::Expr);
}

auto HandleLetAfterPatternAsLet(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal)) {
    context.AddLeafNode(NodeKind::LetInitializer, *equals);
    context.PushState(StateKind::Expr);
  }
}

auto HandleLetAfterPatternAsAssociatedConstant(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  context.AddNode(NodeKind::AssociatedConstantNameAndType, state.token,
                  state.has_error);

  if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal)) {
    context.AddLeafNode(NodeKind::AssociatedConstantInitializer, *equals);
    context.PushState(StateKind::Expr);
  }
}

auto HandleLetFinishAsLet(Context& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    context.DiagnoseExpectedDeclSemi(Lex::TokenKind::Let);
    state.has_error = true;
    end_token = context.SkipPastLikelyEnd(state.token);
  }
  context.AddNode(NodeKind::LetDecl, end_token, state.has_error);
}

auto HandleLetFinishAsAssociatedConstant(Context& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    context.DiagnoseExpectedDeclSemi(Lex::TokenKind::Let);
    state.has_error = true;
    end_token = context.SkipPastLikelyEnd(state.token);
  }
  context.AddNode(NodeKind::AssociatedConstantDecl, end_token, state.has_error);
}

}  // namespace Carbon::Parse
