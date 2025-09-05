// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic.h"
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
    context.emitter().Emit(*context.position(),
                           ExpectedAssociatedConstantIdentifier);
    state.has_error = true;
  }

  auto colon_exclaim = context.ConsumeIf(Lex::TokenKind::ColonExclaim);
  if (identifier && !colon_exclaim) {
    CARBON_DIAGNOSTIC(ExpectedAssociatedConstantColonExclaim, Error,
                      "found runtime binding pattern in associated constant "
                      "declaration; expected a `:!` binding");
    context.emitter().Emit(*context.position(),
                           ExpectedAssociatedConstantColonExclaim);
    state.has_error = true;
  }

  if (!identifier || !colon_exclaim) {
    auto end_token = context.SkipPastLikelyEnd(*(context.position() - 1));
    context.AddNode(NodeKind::AssociatedConstantDecl, end_token,
                    /*has_error=*/true);
    state.has_error = true;
    return;
  }

  context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
  state.token = *colon_exclaim;
  context.PushState(state, StateKind::LetFinishAsAssociatedConstant);
  context.PushState(state, StateKind::LetAfterPatternAsAssociatedConstant);
  context.PushState(StateKind::Expr);
}

static auto HandleLetAfterPattern(Context& context, NodeKind init_kind)
    -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal)) {
    context.AddLeafNode(init_kind, *equals);
    context.PushState(StateKind::Expr);
  }
}

auto HandleLetAfterPatternAsRegular(Context& context) -> void {
  HandleLetAfterPattern(context, NodeKind::LetInitializer);
}

auto HandleLetAfterPatternAsAssociatedConstant(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::AssociatedConstantNameAndType, state.token,
                  state.has_error);
  context.PushState(state);
  HandleLetAfterPattern(context, NodeKind::AssociatedConstantInitializer);
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
