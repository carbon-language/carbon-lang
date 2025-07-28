// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

auto HandleLet(Context& context) -> void {
  auto state = context.PopState();

  // These will start at the `let`.
  context.PushState(state, StateKind::LetFinishAsRegular);
  context.PushState(state, StateKind::LetAfterPatternAsRegular);

  // This will start at the pattern.
  context.PushState(StateKind::Pattern);
}

auto HandleAssociatedConstant(Context& context) -> void {
  auto state = context.PopState();

  // Parse the associated constant pattern: identifier :! type
  auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier);
  if (!identifier) {
    CARBON_DIAGNOSTIC(ExpectedAssociatedConstantIdentifier, Error,
                      "expected identifier in associated constant declaration");
    context.emitter().Emit(*context.position(), ExpectedAssociatedConstantIdentifier);
    state.has_error = true;
  }
  
  auto colon_exclaim = context.ConsumeIf(Lex::TokenKind::ColonExclaim);
  if (identifier && !colon_exclaim) {
    CARBON_DIAGNOSTIC(ExpectedAssociatedConstantColonExclaim, Error,
                      "expected `:!` in associated constant declaration");
    context.emitter().Emit(*context.position(), ExpectedAssociatedConstantColonExclaim);
    state.has_error = true;
  }
  
  if (!identifier || !colon_exclaim) {
    // Skip to the end and let the error recovery handle it
    auto end_token = context.SkipPastLikelyEnd(*(context.position() - 1));
    context.AddLeafNode(NodeKind::EmptyDecl, end_token, /*has_error=*/true);
    context.PopAndDiscardState();
    return;
  }
  
  context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
  state.token = *colon_exclaim;
  context.PushState(state, StateKind::LetFinishAsAssociatedConstant);
  context.PushState(state, StateKind::LetAfterPatternAsAssociatedConstant);
  context.PushState(StateKind::Expr);
}

static auto HandleLetAfterPattern(Context& context, NodeKind pattern_kind, NodeKind init_kind) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  context.AddNode(pattern_kind, state.token, state.has_error);

  if (context.PositionIs(Lex::TokenKind::Equal)) {
    context.AddLeafNode(init_kind, context.ConsumeChecked(Lex::TokenKind::Equal));
    context.PushState(StateKind::Expr);
  }
}

auto HandleLetAfterPatternAsRegular(Context& context) -> void {
  HandleLetAfterPattern(context, NodeKind::LetPattern, NodeKind::LetInitializer);
}

auto HandleLetAfterPatternAsAssociatedConstant(Context& context) -> void {
  HandleLetAfterPattern(context, NodeKind::AssociatedConstantNameAndType, NodeKind::AssociatedConstantInitializer);
}

static auto HandleLetFinish(Context& context, NodeKind node_kind) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    context.DiagnoseExpectedDeclSemi(Lex::TokenKind::Let);
    state.has_error = true;
    end_token = context.SkipPastLikelyEnd(state.token);
  }
  context.AddNode(node_kind, end_token, state.has_error);
}

auto HandleLetFinishAsRegular(Context& context) -> void {
  HandleLetFinish(context, NodeKind::LetDecl);
}

auto HandleLetFinishAsAssociatedConstant(Context& context) -> void {
  HandleLetFinish(context, NodeKind::AssociatedConstantDecl);
}

}  // namespace Carbon::Parse
