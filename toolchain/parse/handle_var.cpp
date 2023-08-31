// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles VarAs(Semicolon|For).
static auto HandleVar(Context& context, State finish_state) -> void {
  context.PopAndDiscardState();

  // These will start at the `var`.
  context.PushState(finish_state);
  context.PushState(State::VarAfterPattern);

  context.AddLeafNode(NodeKind::VariableIntroducer, context.Consume());

  // This will start at the pattern.
  context.PushState(State::PatternAsVariable);
}

auto HandleVarAsSemicolon(Context& context) -> void {
  HandleVar(context, State::VarFinishAsSemicolon);
}

auto HandleVarAsFor(Context& context) -> void {
  HandleVar(context, State::VarFinishAsFor);
}

auto HandleVarAfterPattern(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal)) {
    context.AddLeafNode(NodeKind::VariableInitializer, *equals);
    context.PushState(State::Expression);
  }
}

auto HandleVarFinishAsSemicolon(Context& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    // TODO: Disambiguate between statement and member declaration.
    context.EmitExpectedDeclarationSemi(Lex::TokenKind::Var);
    state.has_error = true;
    if (auto semi_token = context.SkipPastLikelyEnd(state.token)) {
      end_token = *semi_token;
    }
  }
  context.AddNode(NodeKind::VariableDeclaration, end_token, state.subtree_start,
                  state.has_error);
}

auto HandleVarFinishAsFor(Context& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::In)) {
    end_token = context.Consume();
  } else if (context.PositionIs(Lex::TokenKind::Colon)) {
    CARBON_DIAGNOSTIC(ExpectedInNotColon, Error,
                      "`:` should be replaced by `in`.");
    context.emitter().Emit(*context.position(), ExpectedInNotColon);
    state.has_error = true;
    end_token = context.Consume();
  } else {
    CARBON_DIAGNOSTIC(ExpectedIn, Error,
                      "Expected `in` after loop `var` declaration.");
    context.emitter().Emit(*context.position(), ExpectedIn);
    state.has_error = true;
  }

  context.AddNode(NodeKind::ForIn, end_token, state.subtree_start,
                  state.has_error);
}

}  // namespace Carbon::Parse
