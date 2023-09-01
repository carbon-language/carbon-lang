// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleParenExpression(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  context.AddLeafNode(NodeKind::ParenExpressionOrTupleLiteralStart,
                      context.ConsumeChecked(Lex::TokenKind::OpenParen));

  if (context.PositionIs(Lex::TokenKind::CloseParen)) {
    state.state = State::ParenExpressionFinishAsTuple;
    context.PushState(state);
  } else {
    state.state = State::ParenExpressionFinishAsNormal;
    context.PushState(state);
    context.PushState(State::ParenExpressionParameterFinishAsUnknown);
    context.PushState(State::Expression);
  }
}

// Handles ParenExpressionParameterFinishAs(Unknown|Tuple).
static auto HandleParenExpressionParameterFinish(Context& context,
                                                 bool as_tuple) -> void {
  auto state = context.PopState();

  auto list_token_kind = context.ConsumeListToken(
      NodeKind::TupleLiteralComma, Lex::TokenKind::CloseParen, state.has_error);
  if (list_token_kind == Context::ListTokenKind::Close) {
    return;
  }

  // If this is the first item and a comma was found, switch to tuple handling.
  // Note this could be `(expr,)` so we may not reuse the current state, but
  // it's still necessary to switch the parent.
  if (!as_tuple) {
    state.state = State::ParenExpressionParameterFinishAsTuple;

    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state == State::ParenExpressionFinishAsNormal)
        << "Unexpected parent state, found: " << finish_state.state;
    finish_state.state = State::ParenExpressionFinishAsTuple;
    context.PushState(finish_state);
  }

  // On a comma, push another expression handler.
  if (list_token_kind == Context::ListTokenKind::Comma) {
    context.PushState(state);
    context.PushState(State::Expression);
  }
}

auto HandleParenExpressionParameterFinishAsUnknown(Context& context) -> void {
  HandleParenExpressionParameterFinish(context, /*as_tuple=*/false);
}

auto HandleParenExpressionParameterFinishAsTuple(Context& context) -> void {
  HandleParenExpressionParameterFinish(context, /*as_tuple=*/true);
}

auto HandleParenExpressionFinishAsNormal(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::ParenExpression, context.Consume(),
                  state.subtree_start, state.has_error);
}

auto HandleParenExpressionFinishAsTuple(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::TupleLiteral, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
