// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleCallExpression(Context& context) -> void {
  auto state = context.PopState();

  state.state = State::CallExpressionFinish;
  context.PushState(state);

  context.AddNode(NodeKind::CallExpressionStart, context.Consume(),
                  state.subtree_start, state.has_error);
  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    context.PushState(State::CallExpressionParameterFinish);
    context.PushState(State::Expression);
  }
}

auto HandleCallExpressionParameterFinish(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(NodeKind::CallExpressionComma,
                               Lex::TokenKind::CloseParen, state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(State::CallExpressionParameterFinish);
    context.PushState(State::Expression);
  }
}

auto HandleCallExpressionFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::CallExpression, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
