// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleCallExpr(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::CallExprFinish);

  context.AddNode(NodeKind::CallExprStart, context.Consume(),
                  state.subtree_start, state.has_error);
  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    context.PushState(State::CallExprParamFinish);
    context.PushState(State::Expr);
  }
}

auto HandleCallExprParamFinish(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(NodeKind::CallExprComma,
                               Lex::TokenKind::CloseParen, state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(State::CallExprParamFinish);
    context.PushState(State::Expr);
  }
}

auto HandleCallExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::CallExpr, context.Consume(), state.subtree_start,
                  state.has_error);
}

}  // namespace Carbon::Parse
