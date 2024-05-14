// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleOnlyParenExpr(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  auto open_paren = context.ConsumeChecked(Lex::TokenKind::OpenParen);
  context.AddLeafNode(NodeKind::ParenExprStart, open_paren);

  state.token = open_paren;
  context.PushState(state, State::OnlyParenExprFinish);
  context.PushState(State::Expr);
}

static auto FinishParenExpr(Context& context,
                            const Context::StateStackEntry& state) -> void {
  context.AddNode(NodeKind::ParenExpr, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleOnlyParenExprFinish(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(UnexpectedTokenInCompoundMemberAccess, Error,
                        "Expected `)`.");
      context.emitter().Emit(*context.position(),
                             UnexpectedTokenInCompoundMemberAccess);
      state.has_error = true;
    }

    // Recover from the invalid token.
    context.SkipTo(context.tokens().GetMatchedClosingToken(state.token));
  }

  FinishParenExpr(context, state);
}

auto HandleParenExpr(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren. The placeholder will be replaced at the end
  // based on whether we determine this is a tuple or parenthesized expression.
  context.AddLeafNode(NodeKind::Placeholder,
                      context.ConsumeChecked(Lex::TokenKind::OpenParen));

  if (context.PositionIs(Lex::TokenKind::CloseParen)) {
    context.PushState(state, State::TupleLiteralFinish);
  } else {
    context.PushState(state, State::ParenExprFinish);
    context.PushState(State::ExprAfterOpenParenFinish);
    context.PushState(State::Expr);
  }
}

auto HandleExprAfterOpenParenFinish(Context& context) -> void {
  auto state = context.PopState();

  auto list_token_kind = context.ConsumeListToken(
      NodeKind::TupleLiteralComma, Lex::TokenKind::CloseParen, state.has_error);
  if (list_token_kind == Context::ListTokenKind::Close) {
    return;
  }

  // We found a comma, so switch parent state to tuple handling.
  auto finish_state = context.PopState();
  CARBON_CHECK(finish_state.state == State::ParenExprFinish)
      << "Unexpected parent state, found: " << finish_state.state;
  context.PushState(finish_state, State::TupleLiteralFinish);

  // If the comma is not immediately followed by a close paren, push handlers
  // for the next tuple element.
  if (list_token_kind != Context::ListTokenKind::CommaClose) {
    context.PushState(state, State::TupleLiteralElementFinish);
    context.PushState(State::Expr);
  }
}

auto HandleTupleLiteralElementFinish(Context& context) -> void {
  auto state = context.PopState();

  if (context.ConsumeListToken(NodeKind::TupleLiteralComma,
                               Lex::TokenKind::CloseParen, state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(state);
    context.PushState(State::Expr);
  }
}

auto HandleParenExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.ReplacePlaceholderNode(state.subtree_start, NodeKind::ParenExprStart,
                                 state.token);
  FinishParenExpr(context, state);
}

auto HandleTupleLiteralFinish(Context& context) -> void {
  auto state = context.PopState();

  context.ReplacePlaceholderNode(state.subtree_start,
                                 NodeKind::TupleLiteralStart, state.token);
  context.AddNode(NodeKind::TupleLiteral, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
