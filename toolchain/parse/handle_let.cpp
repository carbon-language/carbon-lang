// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleLet(Context& context) -> void {
  auto state = context.PopState();

  // These will start at the `let`.
  context.PushState(state, State::LetFinish);
  context.PushState(state, State::LetAfterPattern);

  // This will start at the pattern.
  context.PushState(State::Pattern);
}

auto HandleLetAfterPattern(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal)) {
    context.AddLeafNode(NodeKind::LetInitializer, *equals);
    context.PushState(State::Expr);
  } else if (!state.has_error) {
    CARBON_DIAGNOSTIC(
        ExpectedInitializerAfterLet, Error,
        "Expected `=`; `let` declaration must have an initializer.");
    context.emitter().Emit(*context.position(), ExpectedInitializerAfterLet);
    context.ReturnErrorOnState();
  }
}

auto HandleLetFinish(Context& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    context.EmitExpectedDeclSemi(Lex::TokenKind::Let);
    state.has_error = true;
    if (auto semi_token = context.SkipPastLikelyEnd(state.token)) {
      end_token = *semi_token;
    }
  }
  context.AddNode(NodeKind::LetDecl, end_token, state.subtree_start,
                  state.has_error);
}

}  // namespace Carbon::Parse
