// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleOnlyParenExpr(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  auto open_paren = context.ConsumeChecked(Lex::TokenKind::OpenParen);
  context.AddLeafNode(NodeKind::ParenExprStart, open_paren);

  state.token = open_paren;
  context.PushState(state, StateKind::OnlyParenExprFinish);
  context.PushState(StateKind::Expr);
}

static auto FinishParenExpr(Context& context, const Context::State& state)
    -> void {
  context.AddNode(NodeKind::ParenExpr, context.Consume(), state.has_error);
}

auto HandleOnlyParenExprFinish(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(UnexpectedTokenInCompoundMemberAccess, Error,
                        "expected `)`");
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
    context.PushState(state, StateKind::TupleLiteralFinish);
  } else {
    if (context.PositionIs(Lex::TokenKind::Ref)) {
      context.PushState(state, StateKind::ParenExprFinishAsRef);
      context.PushState(StateKind::ExprAfterOpenParenFinish);
      context.PushState(StateKind::RefTagFinish);
      context.ConsumeChecked(Lex::TokenKind::Ref);
    } else {
      context.PushState(state, StateKind::ParenExprFinishAsRegular);
      context.PushState(StateKind::ExprAfterOpenParenFinish);
    }
    context.PushState(StateKind::Expr);
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
  CARBON_CHECK(finish_state.kind == StateKind::ParenExprFinishAsRef ||
                   finish_state.kind == StateKind::ParenExprFinishAsRegular,
               "Unexpected parent state, found: {0}", finish_state.kind);
  context.PushState(finish_state, StateKind::TupleLiteralFinish);

  // If the comma is not immediately followed by a close paren, push handlers
  // for the next tuple element.
  if (list_token_kind != Context::ListTokenKind::CommaClose) {
    context.PushState(state, StateKind::TupleLiteralElementFinish);
    if (context.PositionIs(Lex::TokenKind::Ref)) {
      context.PushState(StateKind::RefTagFinish);
      context.ConsumeChecked(Lex::TokenKind::Ref);
    }
    context.PushState(StateKind::Expr);
  }
}

auto HandleTupleLiteralElementFinish(Context& context) -> void {
  auto state = context.PopState();

  if (context.ConsumeListToken(NodeKind::TupleLiteralComma,
                               Lex::TokenKind::CloseParen, state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(state);
    if (context.PositionIs(Lex::TokenKind::Ref)) {
      context.PushState(StateKind::RefTagFinish);
      context.ConsumeChecked(Lex::TokenKind::Ref);
    }
    context.PushState(StateKind::Expr);
  }
}

static auto HandleParenExprFinish(Context& context, StateKind state_kind)
    -> void {
  auto state = context.PopState();
  if (state_kind == StateKind::ParenExprFinishAsRef) {
    Lex::TokenIndex ref_position(state.token.index + 1);
    CARBON_DIAGNOSTIC(UnexpectedRef, Error,
                      "found `ref` in unexpected position");
    context.emitter().Emit(ref_position, UnexpectedRef);
  }
  context.ReplacePlaceholderNode(state.subtree_start, NodeKind::ParenExprStart,
                                 state.token,
                                 state_kind == StateKind::ParenExprFinishAsRef);
  FinishParenExpr(context, state);
}

auto HandleParenExprFinishAsRegular(Context& context) -> void {
  HandleParenExprFinish(context, StateKind::ParenExprFinishAsRegular);
}

auto HandleParenExprFinishAsRef(Context& context) -> void {
  HandleParenExprFinish(context, StateKind::ParenExprFinishAsRef);
}

auto HandleTupleLiteralFinish(Context& context) -> void {
  auto state = context.PopState();

  context.ReplacePlaceholderNode(state.subtree_start,
                                 NodeKind::TupleLiteralStart, state.token);
  context.AddNode(NodeKind::TupleLiteral, context.Consume(), state.has_error);
}

}  // namespace Carbon::Parse
