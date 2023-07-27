// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleIndexExpression(ParserContext& context) -> void {
  auto state = context.PopState();
  state.state = ParserState::IndexExpressionFinish;
  context.PushState(state);
  context.AddNode(ParseNodeKind::IndexExpressionStart, context.Consume(),
                  state.subtree_start, state.has_error);
  if (context.PositionIs(TokenKind::CloseSquareBracket)) {
    CARBON_DIAGNOSTIC(InvalidIndexExpression, Error, "Expected integer.");
    context.emitter().Emit(*context.position(), InvalidIndexExpression);
    context.ReturnErrorOnState();
  } else {
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleIndexExpressionFinish(ParserContext& context) -> void {
  auto state = context.PopState();
  context.AddNode(ParseNodeKind::IndexExpression, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
