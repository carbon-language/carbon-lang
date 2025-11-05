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

static auto HandleRefTagFinish(Context& context, StateKind state_kind) -> void {
  auto state = context.PopState();
  if (state_kind == StateKind::RefTagFinishAsAfterOpenParen &&
      !context.PositionIs(Lex::TokenKind::Comma)) {
    CARBON_DIAGNOSTIC(UnexpectedRef, Error,
                      "found `ref` in unexpected position");
    context.emitter().Emit(state.token, UnexpectedRef);
  }
  context.AddNode(NodeKind::RefTag, state.token, state.has_error);
}

auto HandleRefTagFinishAsRegular(Context& context) -> void {
  HandleRefTagFinish(context, StateKind::RefTagFinishAsRegular);
}

auto HandleRefTagFinishAsAfterOpenParen(Context& context) -> void {
  HandleRefTagFinish(context, StateKind::RefTagFinishAsAfterOpenParen);
}

static auto StartTupleLiteralElement(Context& context) -> void {
  context.PushState(StateKind::TupleLiteralElementFinish);
  if (context.PositionIs(Lex::TokenKind::Ref)) {
    context.PushState(StateKind::RefTagFinishAsRegular);
    context.ConsumeChecked(Lex::TokenKind::Ref);
  }
  context.PushState(StateKind::Expr);
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
    context.PushState(state, StateKind::ParenExprFinish);
    context.PushState(StateKind::ExprAfterOpenParenFinish);
    if (context.PositionIs(Lex::TokenKind::Ref)) {
      context.PushState(StateKind::RefTagFinishAsAfterOpenParen);
      context.ConsumeChecked(Lex::TokenKind::Ref);
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
  CARBON_CHECK(finish_state.kind == StateKind::ParenExprFinish,
               "Unexpected parent state, found: {0}", finish_state.kind);
  context.PushState(finish_state, StateKind::TupleLiteralFinish);

  // If the comma is not immediately followed by a close paren, push handlers
  // for the next tuple element.
  if (list_token_kind != Context::ListTokenKind::CommaClose) {
    StartTupleLiteralElement(context);
  }
}

auto HandleTupleLiteralElementFinish(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(NodeKind::TupleLiteralComma,
                               Lex::TokenKind::CloseParen, state.has_error) ==
      Context::ListTokenKind::Comma) {
    StartTupleLiteralElement(context);
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
  context.AddNode(NodeKind::TupleLiteral, context.Consume(), state.has_error);
}

auto HandleCallExpr(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, StateKind::CallExprFinish);

  context.AddNode(NodeKind::CallExprStart, context.Consume(), state.has_error);
  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    StartTupleLiteralElement(context);
  }
}

auto HandleCallExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::CallExpr, context.Consume(), state.has_error);
}

}  // namespace Carbon::Parse
