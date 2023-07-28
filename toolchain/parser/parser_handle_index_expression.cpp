// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/token_kind.h"
#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleIndexExpression(ParserContext& context) -> void {
  auto state = context.PopState();
  state.state = ParserState::IndexExpressionFinish;
  context.PushState(state);
  context.AddNode(ParseNodeKind::IndexExpressionStart,
                  context.ConsumeChecked(TokenKind::OpenSquareBracket),
                  state.subtree_start, state.has_error);
  if (context.PositionIs(TokenKind::CloseSquareBracket)) {
    CARBON_DIAGNOSTIC(ExpectedExpressionInside, Error,
                      "Expected expression inside the square bracket.", );
    context.emitter().Emit(*context.position(), ExpectedExpressionInside);
    context.ReturnErrorOnState();
  } else {
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleIndexExpressionFinish(ParserContext& context) -> void {
  auto state = context.PopState();
  context.AddNode(ParseNodeKind::IndexExpression,
                  context.ConsumeChecked(TokenKind::CloseSquareBracket),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
