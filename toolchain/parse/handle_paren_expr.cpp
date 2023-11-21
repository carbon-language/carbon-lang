// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleParenExprOrTupleLiteral(Context& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  context.AddLeafNode(NodeKind::ParenExprOrTupleLiteralStart,
                      context.ConsumeChecked(Lex::TokenKind::OpenParen));

  if (context.PositionIs(Lex::TokenKind::CloseParen)) {
    state.state = State::ParenExprOrTupleLiteralFinishAsTuple;
    context.PushState(state);
  } else {
    state.state = State::ParenExprOrTupleLiteralFinishAsParenExpr;
    context.PushState(state);
    context.PushState(State::ParenExprOrTupleLiteralParamFinishAsUnknown);
    context.PushState(State::Expr);
  }
}

// Handles ParenExprParamFinishAs(Unknown|Tuple).
static auto HandleParenExprOrTupleLiteralParamFinish(Context& context,
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
    state.state = State::ParenExprOrTupleLiteralParamFinishAsTuple;

    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state ==
                 State::ParenExprOrTupleLiteralFinishAsParenExpr)
        << "Unexpected parent state, found: " << finish_state.state;
    finish_state.state = State::ParenExprOrTupleLiteralFinishAsTuple;
    context.PushState(finish_state);
  }

  // On a comma, push another expression handler.
  if (list_token_kind == Context::ListTokenKind::Comma) {
    context.PushState(state);
    context.PushState(State::Expr);
  }
}

auto HandleParenExprOrTupleLiteralParamFinishAsUnknown(Context& context)
    -> void {
  HandleParenExprOrTupleLiteralParamFinish(context, /*as_tuple=*/false);
}

auto HandleParenExprOrTupleLiteralParamFinishAsTuple(Context& context) -> void {
  HandleParenExprOrTupleLiteralParamFinish(context, /*as_tuple=*/true);
}

auto HandleParenExprOrTupleLiteralFinishAsParenExpr(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::ParenExpr, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleParenExprOrTupleLiteralFinishAsTuple(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::TupleLiteral, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
