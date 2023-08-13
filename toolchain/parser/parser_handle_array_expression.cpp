// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parser_context.h"
#include "toolchain/parser/parser_state.h"

namespace Carbon {

auto ParserHandleArrayExpression(ParserContext& context) -> void {
  auto state = context.PopState();
  state.state = ParserState::ArrayExpressionSemi;
  context.AddLeafNode(ParseNodeKind::ArrayExpressionStart,
                      context.ConsumeChecked(TokenKind::OpenSquareBracket),
                      state.has_error);
  context.PushState(state);
  context.PushState(ParserState::Expression);
}

auto ParserHandleArrayExpressionSemi(ParserContext& context) -> void {
  auto state = context.PopState();
  auto semi = context.ConsumeIf(TokenKind::Semi);
  if (!semi) {
    context.AddNode(ParseNodeKind::ArrayExpressionSemi, *context.position(),
                    state.subtree_start, true);
    CARBON_DIAGNOSTIC(ExpectedArraySemi, Error, "Expected `;` in array type.");
    context.emitter().Emit(*context.position(), ExpectedArraySemi);
    state.has_error = true;
  } else {
    context.AddNode(ParseNodeKind::ArrayExpressionSemi, *semi,
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
  context.ConsumeAndAddCloseSymbol(
      *(TokenizedBuffer::TokenIterator(state.token)), state,
      ParseNodeKind::ArrayExpression);
}

}  // namespace Carbon
