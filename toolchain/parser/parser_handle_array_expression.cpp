// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/token_kind.h"
#include "toolchain/parser/parser_context.h"
#include "toolchain/parser/parser_state.h"

namespace Carbon {

auto ParserHandleArrayExpression(ParserContext& context) -> void {
  auto state = context.PopState();
  state.state = ParserState::ArrayExpressionSemi;
  context.AddNode(ParseNodeKind::ArrayExpressionStart,
                  context.ConsumeChecked(TokenKind::OpenSquareBracket),
                  state.subtree_start, state.has_error);
  context.PushState(state);
  context.PushState(ParserState::Expression);
}

auto ParserHandleArrayExpressionSemi(ParserContext& context) -> void {
  auto state = context.PopState();
  if (context.PositionKind() != TokenKind::Semi) {
    context.AddNode(ParseNodeKind::ArrayExpressionSemi, *context.position(),
                    state.subtree_start, true);
    CARBON_DIAGNOSTIC(ExpectedArraySemi, Error,
                      "Invalid array declaration. Expected Semi.");
    context.emitter().Emit(*context.position(), ExpectedArraySemi);
    context.ReturnErrorOnState();
  } else {
    context.AddNode(ParseNodeKind::ArrayExpressionSemi,
                    context.ConsumeChecked(TokenKind::Semi),
                    state.subtree_start, state.has_error);
  }
  state.state = ParserState::ArrayExpressionFinish;
  context.PushState(state);

  if (!context.PositionIs(TokenKind::CloseSquareBracket)) {
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleArrayExpressionFinish(ParserContext& context) -> void {
  auto state = context.PopState();
  context.AddNode(ParseNodeKind::ArrayExpression,
                  context.ConsumeChecked(TokenKind::CloseSquareBracket),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
