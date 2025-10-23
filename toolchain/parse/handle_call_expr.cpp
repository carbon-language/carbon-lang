// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleRefTagFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::RefTag, state.token, state.has_error);
}

auto HandleCallExpr(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, StateKind::CallExprFinish);

  context.AddNode(NodeKind::CallExprStart, context.Consume(), state.has_error);
  if (!context.PositionIs(Lex::TokenKind::CloseParen)) {
    context.PushState(StateKind::CallExprParamFinish);
    if (context.PositionIs(Lex::TokenKind::Ref)) {
      context.PushState(StateKind::RefTagFinish);
      context.ConsumeChecked(Lex::TokenKind::Ref);
    }
    context.PushState(StateKind::Expr);
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
    context.PushState(StateKind::CallExprParamFinish);
    if (context.PositionIs(Lex::TokenKind::Ref)) {
      context.PushState(StateKind::RefTagFinish);
      context.ConsumeChecked(Lex::TokenKind::Ref);
    }
    context.PushState(StateKind::Expr);
  }
}

auto HandleCallExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::CallExpr, context.Consume(), state.has_error);
}

}  // namespace Carbon::Parse
