// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleParenExpression(ParserContext& context) -> void {
  auto state = context.PopState();

  // Advance past the open paren.
  context.AddLeafNode(ParseNodeKind::ParenExpressionOrTupleLiteralStart,
                      context.ConsumeChecked(TokenKind::OpenParen));

  if (context.PositionIs(TokenKind::CloseParen)) {
    state.state = ParserState::ParenExpressionFinishAsTuple;
    context.PushState(state);
  } else {
    state.state = ParserState::ParenExpressionFinishAsNormal;
    context.PushState(state);
    context.PushState(ParserState::ParenExpressionParameterFinishAsUnknown);
    context.PushState(ParserState::Expression);
  }
}

// Handles ParenExpressionParameterFinishAs(Unknown|Tuple).
static auto ParserHandleParenExpressionParameterFinish(ParserContext& context,
                                                       bool as_tuple) -> void {
  auto state = context.PopState();

  auto list_token_kind = context.ConsumeListToken(
      ParseNodeKind::TupleLiteralComma, TokenKind::CloseParen, state.has_error);
  if (list_token_kind == ParserContext::ListTokenKind::Close) {
    return;
  }

  // If this is the first item and a comma was found, switch to tuple handling.
  // Note this could be `(expr,)` so we may not reuse the current state, but
  // it's still necessary to switch the parent.
  if (!as_tuple) {
    state.state = ParserState::ParenExpressionParameterFinishAsTuple;

    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state ==
                 ParserState::ParenExpressionFinishAsNormal)
        << "Unexpected parent state, found: " << finish_state.state;
    finish_state.state = ParserState::ParenExpressionFinishAsTuple;
    context.PushState(finish_state);
  }

  // On a comma, push another expression handler.
  if (list_token_kind == ParserContext::ListTokenKind::Comma) {
    context.PushState(state);
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleParenExpressionParameterFinishAsUnknown(ParserContext& context)
    -> void {
  ParserHandleParenExpressionParameterFinish(context, /*as_tuple=*/false);
}

auto ParserHandleParenExpressionParameterFinishAsTuple(ParserContext& context)
    -> void {
  ParserHandleParenExpressionParameterFinish(context, /*as_tuple=*/true);
}

auto ParserHandleParenExpressionFinishAsNormal(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::ParenExpression, context.Consume(),
                  state.subtree_start, state.has_error);
}

auto ParserHandleParenExpressionFinishAsTuple(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::TupleLiteral, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
