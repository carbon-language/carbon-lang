// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleCallExpression(ParserContext& context) -> void {
  auto state = context.PopState();

  state.state = ParserState::CallExpressionFinish;
  context.PushState(state);

  context.AddNode(ParseNodeKind::CallExpressionStart, context.Consume(),
                  state.subtree_start, state.has_error);
  if (!context.PositionIs(TokenKind::CloseParen)) {
    context.PushState(ParserState::CallExpressionParameterFinish);
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleCallExpressionParameterFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(ParseNodeKind::CallExpressionComma,
                               TokenKind::CloseParen, state.has_error) ==
      ParserContext::ListTokenKind::Comma) {
    context.PushState(ParserState::CallExpressionParameterFinish);
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleCallExpressionFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::CallExpression, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
