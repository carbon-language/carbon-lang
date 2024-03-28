// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleOnlyParenExpr(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  context.AddLeafNode(NodeKind::ExprOpenParen,
                      context.ConsumeChecked(Lex::TokenKind::OpenParen));

  context.PushState(state, State::ParenExprFinish);
  context.PushState(State::OnlyParenExprFinish);
  context.PushState(State::Expr);
}

auto HandleOnlyParenExprFinish(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(UnexpectedTokenInCompoundMemberAccess, Error,
                        "Expected `)`.");
      context.emitter().Emit(*context.position(),
                             UnexpectedTokenInCompoundMemberAccess);
      context.ReturnErrorOnState();
    }

    // Recover from the invalid token.
    auto end_of_element = context.FindNextOf({Lex::TokenKind::CloseParen});
    // The lexer guarantees that parentheses are balanced.
    CARBON_CHECK(end_of_element) << "missing matching `(` for `)`";
    context.SkipTo(*end_of_element);
  }
}

auto HandleParenExpr(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  context.AddLeafNode(NodeKind::ExprOpenParen,
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

  context.AddNode(NodeKind::ParenExpr, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleTupleLiteralFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::TupleLiteral, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
